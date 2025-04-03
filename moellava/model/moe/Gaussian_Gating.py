from deepspeed.utils.timer import SynchronizedWallClockTimer
from deepspeed.utils import logger
from deepspeed.utils.bwc import bwc_tensor_model_parallel_world_size
from typing import Callable, Dict, TYPE_CHECKING, Any, Optional, Tuple, Union
import math

import torch
from torch import Tensor
from torch import nn
from torch.nn import Module
import torch.nn.functional as F
from deepspeed.utils import groups
from deepspeed.moe.mappings import drop_tokens, gather_tokens
from deepspeed.moe.sharded_moe import _capacity, _one_hot_to_float, einsum
from deepspeed import comm as dist

from .down_projector import Down_Projector 
if TYPE_CHECKING:
    Base = Module[Tensor]
else:
    Base = Module

TOPK_GATE_TIMER = 'topk_gate'
MOE_TIMER = 'moe'
FIRST_ALLTOALL_TIMER = '1st_a2a'
SECOND_ALLTOALL_TIMER = '2nd_a2a'

uniform_map: Dict[torch.device, Callable] = {}
gumbel_map: Dict[torch.device, Callable] = {}
exp_selection_uniform_map: Dict[torch.device, Callable] = {}

try:
    from tutel import moe as tutel_moe
    TUTEL_INSTALLED = True
except:
    TUTEL_INSTALLED = False
    pass

def top2gating(
        logits: Tensor,
        capacity_factor: float,
        min_capacity: int,
        drop_tokens: bool = True,
        ep_group: Union[torch.distributed.ProcessGroup, None] = None
    ) -> Tuple[Tensor, Tensor, Tensor, Tensor]:
    """Implements Top2Gating on logits."""
    # everything is in fp32 in this function
    gates = F.softmax(logits, dim=1)

    org_gates = gates.clone().detach()
    # Create a mask for 1st's expert per token
    indices1_s = torch.argmax(gates, dim=1)
    num_experts = int(gates.shape[1])
    mask1 = F.one_hot(indices1_s, num_classes=num_experts)

    # Replace top-expert with min value
    logits_except1 = logits.masked_fill(mask1.bool(), float("-inf"))
    indices2_s = torch.argmax(logits_except1, dim=1)
    mask2 = F.one_hot(indices2_s, num_classes=num_experts)

    # Compute locations in capacity buffer
    locations1 = torch.cumsum(mask1, dim=0) - 1
    locations2 = torch.cumsum(mask2, dim=0) - 1
    # Update 2nd's location by accounting for locations of 1st
    locations2 += torch.sum(mask1, dim=0, keepdim=True)

    # gating decisions
    exp_counts = torch.sum(mask1 + mask2, dim=0).detach().to(logits.device)

    if drop_tokens:
        # Calculate configured capacity and remove locations outside capacity from mask
        capacity = _capacity(gates, torch.tensor(capacity_factor * 2), torch.tensor(min_capacity))
        mask1 *= torch.lt(locations1, capacity)
        mask2 *= torch.lt(locations2, capacity)
    else:
        # Do not drop tokens - set capacity according to current expert assignments
        new_capacity = torch.max(exp_counts)
        if ep_group is not None:
            dist.all_reduce(new_capacity, op=dist.ReduceOp.MAX, group=ep_group)
        if groups._get_expert_model_parallel_world_size() == 1:
            # If the non-expert is tensor-parallel, we need to pad the capacity to 'tp'.
            # This is since we are going to activate drop_tokens() to drop duplicate tokens.
            tp = 1 if groups.mpu is None else bwc_tensor_model_parallel_world_size(mpu=groups.mpu)
            new_capacity = torch.ceil(new_capacity / tp).mul(tp).to(new_capacity.dtype)
        capacity = new_capacity

    # Store the capacity location for each token
    locations1_s = torch.sum(locations1 * mask1, dim=1)
    locations2_s = torch.sum(locations2 * mask2, dim=1)

    # Normalize gate probabilities
    mask1_float = mask1.float()
    mask2_float = mask2.float()
    gates1_s = einsum("se,se->s", gates, mask1_float)
    gates2_s = einsum("se,se->s", gates, mask2_float)
    denom_s = gates1_s + gates2_s
    # Avoid divide-by-zero
    denom_s = torch.clamp(denom_s, min=torch.finfo(denom_s.dtype).eps)
    gates1_s /= denom_s
    gates2_s /= denom_s

    # Calculate combine_weights and dispatch_mask
    gates1 = einsum("s,se->se", gates1_s, mask1_float)
    gates2 = einsum("s,se->se", gates2_s, mask2_float)
    locations1_sc = _one_hot_to_float(locations1_s, capacity)
    locations2_sc = _one_hot_to_float(locations2_s, capacity)
    combine1_sec = einsum("se,sc->sec", gates1, locations1_sc)
    combine2_sec = einsum("se,sc->sec", gates2, locations2_sc)
    combine_weights = combine1_sec + combine2_sec
    dispatch_mask = combine_weights.bool()

    return combine_weights, dispatch_mask, exp_counts, org_gates

class MixGroup(nn.Module):
    def __init__(
        self,
        num_experts: int,        # E: number of experts
        components_per_expert: int = 4,  # C: components per expert
        projection_dim: int = 4,  # New parameter for reduced dimension
        drop_centers:bool = True,
    ):
        super().__init__()

        self.projection_dim = projection_dim
        self.num_experts = num_experts
        self.components_per_expert = components_per_expert
        self.total_components = num_experts * components_per_expert  # E*C: total number of Gaussian components
        self.drop_centers = drop_centers

        self.means = nn.Parameter(torch.randn(self.total_components, projection_dim, dtype=torch.float) / math.sqrt(self.projection_dim))
        self.log_vars = nn.Parameter(torch.ones(self.total_components, self.projection_dim, dtype=torch.float)*4)
        # self.vars = nn.Parameter(torch.ones(self.total_components ,self.projection_dim, dtype = torch.float) * 100)
        self.mix_logits = nn.Parameter(torch.zeros(self.total_components, dtype= torch.float))
        self.soft_plus = nn.Softplus()

    def adaptive_dropout(self):
        with torch.no_grad():
            mix_prob = F.softmax(self.mix_logits)
            # alpha = 1 + (mix_prob * mix_prob.log2()).sum() / math.log2(self.total_components)
            # drop_prob = alpha * mix_prob
            # mask = torch.bernoulli(drop_prob)
            mask = torch.bernoulli(mix_prob)
            return mask.bool()

    def is_dead_center(self):
        with torch.no_grad():
            mix_prob = F.softmax(self.mix_logits)
            return torch.bernoulli(F.relu(-self.total_components * mix_prob + 1)).bool()

    def forward(self, input: torch.Tensor, should_print: bool = False) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            input: [B, D] where B is batch size, D is model dimension
        Returns:
            expert_logits: [B, E] logits for each expert
            gmm_logits: [B, E*C] logits for each component
        """
        
        diff = (input.unsqueeze(1) - self.means.unsqueeze(0))
        # vars = self.soft_plus(self.vars)  # [E*C, D]
        vars = torch.exp(self.log_vars)


        # log_vars = torch.log(vars) # [E*C, D]
        log_det_term = torch.sum(self.log_vars, dim=-1)

        mahalanobis_dist = diff * diff / (vars + 1e-6)

        component_logits = -0.5 * (
            torch.sum(mahalanobis_dist, dim=-1) + 
            log_det_term +      # [E*C]
            self.projection_dim * math.log(2 * math.pi)  # scalar
        )
        
        log_mix_probs = F.log_softmax(self.mix_logits.view(-1), dim=0)  # [E*C]
        gmm_logits = component_logits + log_mix_probs.unsqueeze(0)  # [B, E*C]

        dead_logits = None
        if self.training:
            is_dead_centers = self.is_dead_center()
            if sum(is_dead_centers.int()) >= 2:
                if should_print:
                    print(is_dead_centers)
                dead_log_vars = self.log_vars[is_dead_centers,:]
                dead_det_term = torch.sum(dead_log_vars,dim=-1)
                dead_mdist = mahalanobis_dist[:,is_dead_centers,:]
                dead_component_logits = -0.5 * (
                    dead_mdist.sum(dim=-1) +
                    dead_det_term + 
                    self.projection_dim * math.log(2 * math.pi)
                )
                dead_log_mix_probs = F.log_softmax(self.mix_logits[is_dead_centers])
                dead_logits = dead_component_logits + dead_log_mix_probs.unsqueeze(0)
        
        if self.drop_centers:
            drop_out_mask = self.adaptive_dropout()
            gmm_logits = gmm_logits.masked_fill(drop_out_mask,float("-inf"))

        posterior = F.softmax(gmm_logits,dim=-1)
        expert_posterior, _ = torch.max(posterior.view(-1,self.num_experts,self.components_per_expert), dim=2)  # [B, E]

        nll = -torch.logsumexp(gmm_logits, dim=1).mean()
        if dead_logits is not None:
            dead_nll = -torch.logsumexp(dead_logits, dim=1).mean()
            nll += dead_nll
                
        return expert_posterior, posterior, nll

class GMMGate(nn.Module):
    def __init__(
        self,
        model_dim: int,         # D: dimension of input features
        num_experts: int,        # E: number of experts
        components_per_expert: int = 4,  # C: components per expert
        k: int = 1,
        projection_dim: int = 4,  # New parameter for reduced dimension
        capacity_factor: float = 1.0,
        eval_capacity_factor: float = 1.0,
        min_capacity: int = 8,
        noisy_gate_policy: bool = False, 
        drop_tokens: bool = True,
        use_rts: bool = True,
        ep_group: Union[torch.distributed.ProcessGroup, None] = None,
        top2_2nd_expert_sampling: bool = False,
        ) -> None:
        super().__init__()

        self.model_dim = model_dim
        self.projection_dim = projection_dim
        self.num_experts = num_experts
        self.components_per_expert = components_per_expert
        self.total_components = num_experts * components_per_expert  # E*C: total number of Gaussian components
        self.k = k
        self.capacity_factor = capacity_factor
        self.eval_capacity_factor = eval_capacity_factor
        self.noisy_gate_policy = noisy_gate_policy
        self.min_capacity = min_capacity
        self.ep_group = ep_group
        self.drop_tokens = drop_tokens
        self.use_rts = use_rts
        self.top2_2nd_expert_sampling = top2_2nd_expert_sampling

        # Dimension reduction layers
        self.projection = Down_Projector(
            model_dim= model_dim,
            projection_dim= projection_dim
        )

        self.gmm = MixGroup(
            num_experts= num_experts,
            components_per_expert= components_per_expert,
            projection_dim= projection_dim
        )

        self.timers = SynchronizedWallClockTimer()
        self.wall_clock_breakdown = False
        self.gate_time = 0.0

        torch.set_printoptions(threshold=10_000,linewidth=240)

        self.step_counter = 0
        self.print_interval = 200 
        self.debugging = True

    def should_print(self):
        # Check if this is rank 0
        if not self.training:
            return False

        if dist.is_initialized():
            if dist.get_rank() != 0:
                return False

        if not self.debugging:
            return False
        
        return self.step_counter % self.print_interval == 0

    def _set_ep_group(self, ep_group):
        assert self.ep_group is None, f'Attempting to override an existing ep_group'
        self.ep_group = ep_group

    def forward(self, input: torch.Tensor, used_token: torch.Tensor = None, use_tutel: bool = False):
        if self.wall_clock_breakdown:
            self.timers(TOPK_GATE_TIMER).start()

        projected_input, recover_loss = self.projection(input.detach())
        projected_input = projected_input.float().detach()
        expert_logits, posterior, nll = self.gmm(projected_input,should_print=self.should_print())
        expert_logits = torch.stack([torch.roll(expert_logits, i, dims = -1) for i in range(self.k)]).sum(dim=0)

        if self.should_print():
            with torch.no_grad():
                print(f"Priors:\n{F.softmax(self.gmm.mix_logits.view(-1), dim=0)}")
                print(f"Vars:\n{self.gmm.log_vars.exp().mean(dim=1)}")
                print("Responsibility:")
                print(posterior[:5,:])
                print(posterior[-5:,:])
                print("Gating:")
                print(expert_logits[:5,:])
                print(expert_logits[-5:,:])

        if self.k == 1:
            raise Exception("Top 1 Gating not implemented for Gaussian Mixture Model Based Gating")
        elif self.k == 2:
            combine_weights, dispatch_mask, exp_counts, org_gates = top2gating(
                logits=expert_logits.detach(),
                capacity_factor=self.capacity_factor if self.training else self.eval_capacity_factor,
                min_capacity=self.min_capacity, 
                drop_tokens=self.drop_tokens, 
                ep_group=self.ep_group,
                # top2_2nd_expert_sampling=self.top2_2nd_expert_sampling
            )
            # gate_output[0] = nll + recover_loss
            # gate_output = tuple(gate_output)
        else:
            raise Exception("Top k Gating not implemented for Gaussian Mixture Model Based Gating")

        self.step_counter += 1

        if self.wall_clock_breakdown:
            self.timers(TOPK_GATE_TIMER).stop()
            self.gate_time = self.timers(TOPK_GATE_TIMER).elapsed(reset=False)

        return nll+recover_loss, combine_weights, dispatch_mask, exp_counts ,org_gates