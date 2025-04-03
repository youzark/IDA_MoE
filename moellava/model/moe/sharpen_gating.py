from typing import Callable, Dict, Optional, Tuple, Union
import math

from deepspeed.utils.timer import SynchronizedWallClockTimer
from deepspeed.utils.bwc import bwc_tensor_model_parallel_world_size
from deepspeed import comm as dist
from deepspeed.utils import groups
from deepspeed.moe.sharded_moe import TopKGate
from deepspeed.moe.sharded_moe import gumbel_rsample, _capacity, _one_hot_to_float, einsum

import torch.distributed as dist
import torch
from torch import Tensor, nn

from .down_projector import Down_Projector 

TOPK_GATE_TIMER = 'topk_gate'

uniform_map: Dict[torch.device, Callable] = {}
exp_selection_uniform_map: Dict[torch.device, Callable] = {}

import torch
import torch.nn.functional as F

@torch.jit.script
def _top_idx(source, k):
    return torch.topk(source, k=k, dim=0)[1]

def sharpen_aux_loss(gates: torch.Tensor) -> torch.Tensor:
    """
    Compute a sharpened auxiliary loss based on KL divergence.
    
    Args:
    logits (torch.Tensor): The raw logits of shape (num_tokens, num_experts)
    
    Returns:
    torch.Tensor: The computed auxiliary loss (scalar)
    """
    
    # Compute f_j (sum over tokens for each expert)
    f_j = torch.sum(gates, dim=0)  # Shape: (num_experts,)
    
    # Compute the numerator of p_ij
    p_ij_numerator = gates**2 / f_j.unsqueeze(0)
    
    # Compute p_ij
    p_ij = p_ij_numerator / torch.sum(p_ij_numerator, dim=-1, keepdim=True)
    
    # Compute KL divergence
    kl_div = torch.sum(p_ij * (torch.log(p_ij) - torch.log(gates)), dim=-1)

    return torch.mean(kl_div)

def multiplicative_jitter(x, device: torch.device, epsilon=1e-2):
    """
    Modified from switch transformer paper. mesh transformers
    Multiply values by a random number between 1-epsilon and 1+epsilon.
    Makes models more resilient to rounding errors introduced by bfloat16.
    This seems particularly important for logits.
    Args:
        x: a torch.tensor
        device: torch.device
        epsilon: a floating point value
    Returns:
        a jittered x.
    """
    if epsilon == 0:
        return x
    uniform = uniform_map.get(device)
    if uniform is None:
        uniform = torch.distributions.uniform.Uniform(low=torch.tensor(1.0 - epsilon, device=device),
                                                      high=torch.tensor(1.0 + epsilon,
                                                                        device=device)).rsample  # type: ignore
        uniform_map[device] = uniform
    return x * uniform(x.shape)

def top1gating(
    logits: Tensor,
    capacity_factor: float,
    min_capacity: int,
    used_token: Tensor = None,
    noisy_gate_policy: Optional[str] = None,
    drop_tokens: bool = True,
    use_rts: bool = True,
    l_aux_type: str = "load_balancing",
    ep_group: Union[torch.distributed.ProcessGroup, None] = None,
    use_tutel: bool = False
    ) -> Tuple[Tensor, Tensor, Tensor, Tensor]:
    """Implements Top1Gating on logits."""
    if noisy_gate_policy == 'RSample':
        logits_w_noise = logits + gumbel_rsample(logits.shape, device=logits.device)
    # everything is in fp32 in this function
    # print(f"Logits Shape: {logits.shape}")
    # print(logits[:5,:])
    gates = F.softmax(logits, dim=1)
    # print(f"Gate Shape: {gates.shape}")
    # print(gates[:5,:])


    capacity = _capacity(gates, torch.tensor(capacity_factor), torch.tensor(min_capacity))
    # print(f"Capacity {capacity}")

    # Create a mask for 1st's expert per token
    # noisy gating
    indices1_s = torch.argmax(logits_w_noise if noisy_gate_policy == 'RSample' else gates, dim=1)
    # print(f"indices1_s {indices1_s.shape}")
    # print(f"{indices1_s[:5]}")
    num_experts = int(gates.shape[1])
    mask1 = F.one_hot(indices1_s, num_classes=num_experts)
    # print(f"mask1 {mask1.shape}")
    # print(mask1[:5,:])

    # mask only used tokens
    # if used_token is not None:
    #     mask1 = einsum("s,se->se", used_token, mask1)

    # gating decisions
    exp_counts = torch.sum(mask1, dim=0).detach().to('cpu')
    # print(f"exp_counts {exp_counts.shape}")

    # if we don't want to drop any tokens
    if not drop_tokens:
        new_capacity = torch.max(exp_counts).to(logits.device)
        # Communicate across expert processes to pick the maximum capacity.
        if ep_group is not None:
            dist.all_reduce(new_capacity, op=dist.ReduceOp.MAX, group=ep_group)
        if groups._get_expert_model_parallel_world_size() == 1:
            # If the non-expert is tensor-parallel, we need to pad the capacity to 'tp'.
            # This is since we are going to activate drop_tokens() to drop duplicate tokens.
            tp = 1 if groups.mpu is None else bwc_tensor_model_parallel_world_size(mpu=groups.mpu)
            new_capacity = torch.ceil(new_capacity / tp).mul(tp).to(new_capacity.dtype)
        # Make sure the capacity value does not exceed the number of tokens.
        capacity = min(new_capacity, torch.tensor(mask1.size(0)).to(new_capacity.device))

    # Compute l_aux
    if l_aux_type == "load_balancing":
        me = torch.mean(gates, dim=0)
        ce = torch.mean(mask1.float(), dim=0)
        l_aux = torch.sum(me * ce) * num_experts
    elif l_aux_type == "sharpen":
        l_aux = sharpen_aux_loss(gates)
    elif l_aux_type == "gaussian":
        l_aux = None
    else:
        raise Exception(f"Unknown l_aux_type :{l_aux_type}, options:'sharpen','load_balancing'")

    # Random Token Selection
    if use_rts:
        uniform = exp_selection_uniform_map.get(logits.device)
        if uniform is None:
            uniform = torch.distributions.uniform.Uniform(low=torch.tensor(0.0, device=logits.device),
                                                          high=torch.tensor(1.0, device=logits.device)).rsample
            exp_selection_uniform_map[logits.device] = uniform

        mask1_rand = mask1 * uniform(mask1.shape)
    else:
        mask1_rand = mask1

    assert logits.shape[
        0] >= min_capacity, "No. of tokens (batch-size) should be greater than min_capacity. Either set min_capacity to 0 or increase your batch size."

    top_idx = _top_idx(mask1_rand, capacity)
    # print(f"top_idx {top_idx.shape}")
    # print(top_idx[:5,:])
    # print(top_idx[:,:1])

    new_mask1 = mask1 * torch.zeros_like(mask1).scatter_(0, top_idx, 1)
    mask1 = new_mask1
    # print(f"Gate {gates.shape}, capacity {capacity}")
    # print(f"mask1 {mask1.shape}")
    # print(mask1[:5,:])

    if use_tutel:
        # Tutel doesn't support index values masked with zero
        # so we need to replace masked indices with -1
        indices_mask = mask1.sum(dim=1) * num_experts - 1
        indices1_s = torch.min(indices1_s, indices_mask)

    # Compute locations in capacity buffer
    if use_tutel:
        locations1 = tutel_moe.fast_cumsum_sub_one(mask1)
    else:
        locations1 = torch.cumsum(mask1, dim=0) - 1
    # print(f"locations1 {locations1.shape}")
    # print(locations1[:5,:])
    # print(locations1[:,:1])

    if use_tutel:
        gates1_s = (gates * mask1).sum(dim=1)
        locations1_s = torch.sum(locations1 * mask1, dim=1)
        return l_aux, capacity, num_experts, [
            indices1_s,
        ], [
            locations1_s,
        ], [
            gates1_s,
        ], exp_counts

    # Store the capacity location for each token
    locations1_s = torch.sum(locations1 * mask1, dim=1)
    # print("locations1_s = torch.sum(locations1 * mask1, dim=1)")
    # print(f"locations1_s {locations1_s.shape}")
    # print(locations1_s[:5])

    # Normalize gate probabilities
    mask1_float = mask1.float()
    org_gates = gates.clone().detach()
    gates = gates * mask1_float
    # print(f"gates {gates.shape}")
    # print(gates[:5,:])

    locations1_sc = _one_hot_to_float(locations1_s, capacity)
    combine_weights = einsum("se,sc->sec", gates, locations1_sc)
    # print(f"location1_sc {locations1_sc.shape}")
    # print(locations1_sc[0,:])
    # print(locations1_sc[:,0])
    # print(f"combine_weights {combine_weights.shape}")
    # print(combine_weights[0])
    # print(combine_weights[:,0,:])

    dispatch_mask = combine_weights.bool()
    # print(f"dispatch_mask {dispatch_mask.shape}")

    return l_aux, combine_weights, dispatch_mask, exp_counts, org_gates

def top2gating(
    logits: Tensor,
    capacity_factor: float,
    min_capacity: int,
    drop_tokens: bool = True,
    l_aux_type: str = "load_balancing",
    ep_group: Union[torch.distributed.ProcessGroup, None] = None,
    top2_2nd_expert_sampling: bool = True
    ) -> Tuple[Tensor, Tensor, Tensor, Tensor]:
    """Implements Top2Gating on logits."""
    # everything is in fp32 in this function
    # if l_aux_type == "gaussian":
    #     gates = logits / logits.sum(dim = -1, keepdim=True)
    # else:
    #     gates = F.softmax(logits, dim=1)
    gates = F.softmax(logits, dim=1)
    org_gates = gates

    # Create a mask for 1st's expert per token
    indices1_s = torch.argmax(gates, dim=1)
    num_experts = int(gates.shape[1])
    mask1 = F.one_hot(indices1_s, num_classes=num_experts)

    if top2_2nd_expert_sampling:
        # Create a mask for 2nd's expert per token using Gumbel-max trick
        # https://timvieira.github.io/blog/post/2014/07/31/gumbel-max-trick/
        logits += gumbel_rsample(logits.shape, device=logits.device)

    # Replace top-expert with min value
    logits_except1 = logits.masked_fill(mask1.bool(), float("-inf"))
    indices2_s = torch.argmax(logits_except1, dim=1)
    mask2 = F.one_hot(indices2_s, num_classes=num_experts)

    # Compute locations in capacity buffer
    locations1 = torch.cumsum(mask1, dim=0) - 1
    locations2 = torch.cumsum(mask2, dim=0) - 1
    # Update 2nd's location by accounting for locations of 1st
    locations2 += torch.sum(mask1, dim=0, keepdim=True)

    # Compute l_aux
    if l_aux_type == "load_balancing" or l_aux_type == "xMoE":
        me = torch.mean(gates, dim=0)
        ce = torch.mean(mask1.float(), dim=0)
        l_aux = torch.sum(me * ce) * num_experts * num_experts
    elif l_aux_type == "sharpen":
        l_aux = sharpen_aux_loss(gates)
    elif l_aux_type == "gaussian":
        l_aux = None
    elif l_aux_type == "aux_free":
        l_aux = None
    else:
        raise Exception(f"Unknown l_aux_type :{l_aux_type}, options:'sharpen','load_balancing'")

    # gating decisions
    exp_counts = torch.sum(mask1 + mask2, dim=0)

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

    return l_aux, combine_weights, dispatch_mask, exp_counts.detach().to('cpu'), org_gates


def topkgating(
    logits: Tensor,
    k: int,
    capacity_factor: float,
    min_capacity: int,
    drop_tokens: bool = True,
    ep_group: Union[torch.distributed.ProcessGroup, None] = None,
    drop_policy: str = "probs",
) -> Tuple[Tensor, Tensor, Tensor, Tensor]:
    """Implements TopKGating on logits."""

    # everything is in fp32 in this function
    # get topk gates
    top_gate, top_idx = torch.topk(logits, k=k, dim=1)
    # gating decisions
    gates = F.softmax(logits, dim=1)
    num_experts = int(gates.shape[1])

    # get topk mask
    topk_masked_gates = torch.zeros_like(logits).scatter(1, top_idx, top_gate)

    mask = torch.zeros_like(gates, dtype=torch.bool).scatter_(1, top_idx, 1)

    exp_counts = torch.sum(mask, dim=0).detach().to(logits.device)

    # Compute l_aux
    me = torch.mean(gates, dim=0)
    ce = torch.mean(mask.float(), dim=0)
    l_aux = torch.mean(me * ce) * num_experts * num_experts / k

    if drop_tokens:
        # Calculate configured capacity and remove locations outside capacity from mask
        capacity = _capacity(gates, torch.tensor(capacity_factor * k), torch.tensor(min_capacity))
        # update mask and locations by capacity

        if drop_policy == 'probs':
            capacity_probs, capacity_indices = torch.topk(topk_masked_gates, k=capacity, dim=0, sorted=False)
            capacity_mask = torch.zeros_like(logits).scatter(0, capacity_indices, 1)
            mask = torch.logical_and(mask, capacity_mask)
            locations = torch.cumsum(mask, dim=0) - 1

        elif drop_policy == "position":
            locations = torch.cumsum(mask, dim=0) - 1
            mask *= torch.lt(locations, capacity)
        else:
            raise ValueError(f"Invalid drop_policy: {drop_policy}")

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

    # normalize gates
    gates_masked = gates * mask
    gates_s = torch.sum(gates_masked, dim=-1, keepdim=True)
    denom_s = torch.clamp(gates_s, min=torch.finfo(gates_masked.dtype).eps)
    gates_masked = gates_masked / denom_s

    # dispatch_mask
    locations_sc = _one_hot_to_float((locations * mask), capacity)

    combine_weights = torch.einsum("se,sec->sec", gates_masked, locations_sc)

    dispatch_mask = combine_weights.bool()

    return l_aux, combine_weights, dispatch_mask, exp_counts

class TopKSharpenGate(TopKGate):
    """Gate module which implements Top2Gating as described in Gshard_ with "Sharpen Aux Loss".
    ::

        gate = TopKGate(model_dim, num_experts)
        l_aux, combine_weights, dispatch_mask = gate(input)

    .. Gshard_: https://arxiv.org/pdf/2006.16668.pdf

    Args:
        model_dim (int):
            size of model embedding dimension
        num_experts (int):
            number of experts in model
    """

    wg: torch.nn.Linear

    def __init__(
        self,
        model_dim: int,
        num_experts: int,
        k: int = 1,
        capacity_factor: float = 1.0,
        eval_capacity_factor: float = 1.0,
        min_capacity: int = 8,
        noisy_gate_policy: Optional[str] = None,
        drop_tokens: bool = True,
        use_rts: bool = True,
        ep_group: Union[torch.distributed.ProcessGroup, None] = None,
        top2_2nd_expert_sampling: bool = True,
        l_aux_type: str = "load_balancing",
        ) -> None:
        super().__init__(model_dim,num_experts,k,capacity_factor,eval_capacity_factor,min_capacity,noisy_gate_policy,drop_tokens,use_rts,ep_group,top2_2nd_expert_sampling)
        self.l_aux_type = l_aux_type

    def forward(
        self,
        input: torch.Tensor,
        used_token: torch.Tensor = None,
        use_tutel: bool = False
        ) -> Tuple[Tensor, Tensor, Tensor]:  # type: ignore

        if self.wall_clock_breakdown:
            self.timers(TOPK_GATE_TIMER).start()

        input_fp32 = input.float()

        # input jittering
        if self.noisy_gate_policy == 'Jitter' and self.training:
            input_fp32 = multiplicative_jitter(input_fp32, device=input.device)

        logits = torch.nn.functional.linear(input_fp32, weight=self.wg.weight.float(), bias=None)

        if self.k == 1:
            gate_output = top1gating(logits, self.capacity_factor if self.training else self.eval_capacity_factor,
                                     self.min_capacity, used_token, self.noisy_gate_policy if self.training else None,
                                     self.drop_tokens, self.use_rts, self.l_aux_type,self.ep_group, use_tutel)

        else:
            gate_output = top2gating(logits, self.capacity_factor if self.training else self.eval_capacity_factor,
                                     self.min_capacity, self.drop_tokens, self.l_aux_type, self.ep_group, self.top2_2nd_expert_sampling)

        if self.wall_clock_breakdown:
            self.timers(TOPK_GATE_TIMER).stop()
            self.gate_time = self.timers(TOPK_GATE_TIMER).elapsed(reset=False)

        return gate_output

# class GMMGate(nn.Module):
#     """Gate module which implements GMM-based gating for MoE.
#     Each expert is modeled by a Gaussian component, and the gating
#     decision is based purely on the Gaussian likelihood.
#     """
#     def __init__(self,
#                  model_dim: int,
#                  num_experts: int,
#                  components_per_expert: int = 4,
#                  k: int = 1,
#                  capacity_factor: float = 1.0,
#                  eval_capacity_factor: float = 1.0,
#                  min_capacity: int = 8,
#                  drop_tokens: bool = True,
#                  use_rts: bool = True,
#                  ep_group: Union[torch.distributed.ProcessGroup, None] = None,
#                  top2_2nd_expert_sampling: bool = True) -> None:
#         super().__init__()

#         self.model_dim = model_dim
#         self.num_experts = num_experts
#         self.components_per_expert = components_per_expert
#         self.total_components = num_experts * components_per_expert
#         self.k = k
#         self.capacity_factor = capacity_factor
#         self.eval_capacity_factor = eval_capacity_factor
#         self.min_capacity = min_capacity
#         self.ep_group = ep_group
#         self.drop_tokens = drop_tokens
#         self.use_rts = use_rts
#         self.top2_2nd_expert_sampling = top2_2nd_expert_sampling

#         # Simple initialization of means and variances
#         self.means = nn.Parameter(torch.randn(self.total_components, model_dim) * 0.01) ## [Expert * Component_count, Hidden]
#         self.log_vars = nn.Parameter(torch.zeros(self.total_components, model_dim)) ## [Expert * Component_count, Hidden]
#         self.mix_logits = nn.Parameter(torch.zeros(num_experts, components_per_expert)) ## [Expert, Component_count]

#         self.timers = SynchronizedWallClockTimer()
#         self.wall_clock_breakdown = False
#         self.gate_time = 0.0
#         self.last_nll = None

#     def compute_logits(self, input: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
#         input = input.float()
#         batch_size = input.size(0)
        
#         # Basic GMM computation without any normalization or clamping
#         diff = (input.unsqueeze(1) - self.means.unsqueeze(0))
#         vars = torch.exp(self.log_vars)
        
#         log_det_term = torch.sum(self.log_vars, dim=-1)
#         mahalanobis_dist = torch.sum(diff * diff / vars.unsqueeze(0), dim=-1)
        
#         component_logits = -0.5 * (
#             mahalanobis_dist +
#             log_det_term +
#             self.model_dim * math.log(2 * math.pi)
#         )
        
#         component_logits = component_logits.view(batch_size, self.num_experts, self.components_per_expert)
#         mix_probs = F.log_softmax(self.mix_logits, dim=1)
        
#         gmm_logits = component_logits + mix_probs.unsqueeze(0)
#         expert_logits = torch.logsumexp(gmm_logits, dim=2)
        
#         return expert_logits, gmm_logits.reshape(batch_size, -1)


#     # def compute_logits(self, input: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
#     #     """Compute both expert logits for gating and full GMM logits for NLL."""
#     #     input = input.float()
#     #     batch_size = input.size(0)
        
#     #     # Clip input values
#     #     input = torch.clamp(input, min=-100, max=100)
        
#     #     # Calculate difference for all components
#     #     diff = (input.unsqueeze(1) - self.means.unsqueeze(0))  # [B, E*C, D]
        
#     #     # Ensure variance doesn't get too small or too large
#     #     log_vars = torch.clamp(self.log_vars, min=-7, max=7)
#     #     vars = torch.exp(log_vars) + 1e-6
        
#     #     # Compute log probability for each component
#     #     log_det_term = torch.sum(log_vars, dim=-1)  # [E*C]
#     #     mahalanobis_dist = torch.sum(diff * diff / vars.unsqueeze(0), dim=-1)  # [B, E*C]
        
#     #     component_logits = -0.5 * (
#     #         mahalanobis_dist +
#     #         log_det_term +
#     #         self.model_dim * math.log(2 * math.pi)
#     #     )  # [B, E*C]
        
#     #     # Reshape to [B, E, C]
#     #     component_logits = component_logits.view(batch_size, self.num_experts, self.components_per_expert)
        
#     #     # Get mixing coefficients per expert
#     #     mix_probs = F.log_softmax(self.mix_logits, dim=1)  # [E, C]
        
#     #     # Compute GMM logits per expert
#     #     gmm_logits = component_logits + mix_probs.unsqueeze(0)  # [B, E, C]
        
#     #     # Compute expert logits by taking logsumexp over components
#     #     expert_logits = torch.logsumexp(gmm_logits, dim=2)  # [B, E]
        
#     #     # Normalize expert logits
#     #     expert_logits = expert_logits - expert_logits.max(dim=1, keepdim=True)[0]
        
#     #     # Compute full GMM logits for NLL
#     #     full_gmm_logits = gmm_logits.reshape(batch_size, -1)  # [B, E*C]
        
#     #     return expert_logits, full_gmm_logits

#     # def compute_logits(self, input: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
#     #     """Compute both expert logits for gating and full GMM logits for NLL."""
#     #     input = input.float()
#     #     batch_size = input.size(0)
        
#     #     # Calculate difference for all components
#     #     diff = (input.unsqueeze(1) - self.means.unsqueeze(0))  # [B, E*C, D]
        
#     #     vars = torch.exp(self.log_vars) + 1e-6
        
#     #     # Compute log probability for each component
#     #     log_det_term = torch.sum(self.log_vars, dim=-1)  # [E*C]
#     #     mahalanobis_dist = torch.sum(diff * diff / vars.unsqueeze(0), dim=-1)  # [B, E*C]
        
#     #     component_logits = -0.5 * (
#     #         mahalanobis_dist +
#     #         log_det_term +
#     #         self.model_dim * math.log(2 * math.pi)
#     #     )  # [B, E*C]
        
#     #     # Reshape to [B, E, C]
#     #     component_logits = component_logits.view(batch_size, self.num_experts, self.components_per_expert)
        
#     #     # Get mixing coefficients per expert
#     #     mix_probs = F.log_softmax(self.mix_logits, dim=1)  # [E, C]
        
#     #     # Compute GMM logits per expert
#     #     gmm_logits = component_logits + mix_probs.unsqueeze(0)  # [B, E, C]
        
#     #     # Compute expert logits by taking logsumexp over components
#     #     expert_logits = torch.logsumexp(gmm_logits, dim=2)  # [B, E]
        
#     #     # Normalize expert logits
#     #     expert_logits = expert_logits - expert_logits.max(dim=1, keepdim=True)[0]
        
#     #     # Compute full GMM logits for NLL
#     #     full_gmm_logits = gmm_logits.reshape(batch_size, -1)  # [B, E*C]
        
#     #     return expert_logits, full_gmm_logits

#     def _set_ep_group(self, ep_group):
#         assert self.ep_group is None, f'Attempting to override an existing ep_group'
#         self.ep_group = ep_group

#     def forward(self,
#                 input: torch.Tensor,
#                 used_token: torch.Tensor = None,
#                 use_tutel: bool = False) -> Tuple[Tensor, Tensor, Tensor, Tensor]:

#         if self.wall_clock_breakdown:
#             self.timers(TOPK_GATE_TIMER).start()

#         # Compute expert logits and GMM logits
#         expert_logits, gmm_logits = self.compute_logits(input)
        
#         # Calculate NLL loss using full GMM
#         self.last_nll = -torch.logsumexp(gmm_logits, dim=-1).mean()

#         # Rest of the forward method remains the same
#         if self.k == 1:
#             gate_output = list(top1gating(expert_logits, 
#                                    self.capacity_factor if self.training else self.eval_capacity_factor,
#                                    self.min_capacity, used_token, None,
#                                    self.drop_tokens, self.use_rts, "gaussian", self.ep_group, use_tutel))
#             gate_output[0] = self.last_nll
#             gate_output = tuple(gate_output)
#         elif self.k == 2:
#             gate_output = list(top2gating(expert_logits,
#                                    self.capacity_factor if self.training else self.eval_capacity_factor,
#                                    self.min_capacity, self.drop_tokens, "gaussian", self.ep_group,
#                                    self.top2_2nd_expert_sampling))
#             gate_output[0] = self.last_nll
#             gate_output = tuple(gate_output)
#         else:
#             gate_output = list(topkgating(expert_logits, self.k,
#                                    self.capacity_factor if self.training else self.eval_capacity_factor,
#                                    self.min_capacity, self.drop_tokens, self.ep_group))
#             gate_output[0] = self.last_nll
#             gate_output = tuple(gate_output)

#         if self.wall_clock_breakdown:
#             self.timers(TOPK_GATE_TIMER).stop()
#             self.gate_time = self.timers(TOPK_GATE_TIMER).elapsed(reset=False)

#         return gate_output

class MixGroup(nn.Module):
    def __init__(
        self,
        num_experts: int,        # E: number of experts
        components_per_expert: int = 4,  # C: components per expert
        projection_dim: int = 4,  # New parameter for reduced dimension
        drop_centers:bool = False,
        group_reactivation:bool = True,
    ):
        super().__init__()

        self.projection_dim = projection_dim
        self.num_experts = num_experts
        self.components_per_expert = components_per_expert
        self.total_components = num_experts * components_per_expert  # E*C: total number of Gaussian components
        self.drop_centers = drop_centers
        self.group_reactivation = group_reactivation

        self.means = nn.Parameter(torch.randn(self.total_components, projection_dim, dtype=torch.float) / math.sqrt(self.projection_dim))
        self.vars = nn.Parameter(torch.ones(self.total_components ,self.projection_dim, dtype = torch.float))
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

    def forward(self, input: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            input: [B, D] where B is batch size, D is model dimension
        Returns:
            expert_logits: [B, E] logits for each expert
            gmm_logits: [B, E*C] logits for each component
        """
        
        diff = (input.unsqueeze(1) - self.means.unsqueeze(0))
        vars = self.soft_plus(self.vars)  # [E*C, D]


        log_vars = torch.log(vars) # [E*C, D]
        log_det_term = torch.sum(log_vars, dim=-1)

        mahalanobis_dist = diff * diff / (vars + 1e-6)

        component_logits = -0.5 * (
            torch.sum(mahalanobis_dist, dim=-1) + 
            log_det_term +      # [E*C]
            self.projection_dim * math.log(2 * math.pi)  # scalar
        )
        
        log_mix_probs = F.log_softmax(self.mix_logits.view(-1), dim=0)  # [E*C]
        gmm_logits = component_logits + log_mix_probs.unsqueeze(0)  # [B, E*C]

        dead_logits = None
        if self.training and self.group_reactivation:
            is_dead_centers = self.is_dead_center()
            if sum(is_dead_centers.int()) >= 2:
                dead_log_vars = log_vars[is_dead_centers,:]
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
                
        return expert_posterior, nll

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
            top2_2nd_expert_sampling: bool = True,
            group_reactivation: bool = True,
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

        self.gmms = nn.ModuleList([
            MixGroup(
                num_experts= num_experts,
                components_per_expert=components_per_expert,
                projection_dim=projection_dim,
                group_reactivation= group_reactivation,
            ) for _ in range(k)
        ])

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
        expert_logits_list, nll_list = zip(*[gmm(projected_input) for gmm in self.gmms])

        
        expert_logits = torch.stack(expert_logits_list).sum(dim=0)

        nll = torch.stack(nll_list).sum(dim=0)
        # gmm_logits, dead_logits,recover_loss= self.compute_logits(input)
        # expert_logits, _ = torch.max(gmm_logits.view(-1,self.num_experts,self.components_per_expert), dim=2)  # [B, E]
        
        # nll = -torch.logsumexp(gmm_logits, dim=1).mean()
        # if dead_logits is not None:
        #     dead_nll = -torch.logsumexp(dead_logits, dim=1).mean()
        #     nll += dead_nll


        if self.should_print():
            with torch.no_grad():
                # print("Logits")
                # print(expert_logits[:5,:])
                print(f"Priors:\n{F.softmax(self.gmms[0].mix_logits.view(-1), dim=0)}")
                print(f"Vars:\n{self.gmms[0].vars.exp().mean(dim=1)}")
                print("Gating:")
                print(expert_logits[:5,:])
                print(expert_logits[-5:,:])

        if self.k == 1:
            gate_output = list(top1gating(
                expert_logits.detach(),
                self.capacity_factor if self.training else self.eval_capacity_factor,
                self.min_capacity,
                used_token,
                None,
                self.drop_tokens,
                self.use_rts,
                "gaussian",
                self.ep_group,
                use_tutel
            ))
            gate_output[0] = nll + recover_loss
            gate_output = tuple(gate_output)
        elif self.k == 2:
            gate_output = list(top2gating(expert_logits.detach(),
                                   self.capacity_factor if self.training else self.eval_capacity_factor,
                                   self.min_capacity, self.drop_tokens, "gaussian", self.ep_group,
                                   self.top2_2nd_expert_sampling))
            gate_output[0] = nll + recover_loss
            gate_output = tuple(gate_output)
        else:
            gate_output = list(topkgating(expert_logits.detach(), self.k,
                                   self.capacity_factor if self.training else self.eval_capacity_factor,
                                   self.min_capacity, self.drop_tokens, self.ep_group))
            gate_output[0] = nll + recover_loss
            gate_output = tuple(gate_output)
        self.step_counter += 1

        if self.wall_clock_breakdown:
            self.timers(TOPK_GATE_TIMER).stop()
            self.gate_time = self.timers(TOPK_GATE_TIMER).elapsed(reset=False)

        return gate_output

        # if self.should_print():
        #     print("$"*100)
        #     print("Vars mean")
        #     print(vars.mean(dim=-1))
        #     print("logs Vars Sum")
        #     print(log_det_term)
        #     print("Diff")
        #     print((diff*diff).sum(-1)[:5,:])
        #     print("MDist")
        #     print(mahalanobis_dist.sum(dim=-1)[:5,:])

        #     def inspect_means_dist():
        #         with torch.no_grad():
        #             mean_diff = self.means.unsqueeze(dim=0) - self.means.unsqueeze(dim=1)
        #             mean_mdist = (mean_diff * mean_diff).sum(dim=-1)
        #             print("Distance between centers")
        #             print(mean_mdist)

        #     inspect_means_dist()

        #     print("Posterior")
        #     posterior = F.softmax(gmm_logits, dim = -1)
        #     print(posterior[:5,:])
        #     print(posterior.sum(dim=0).to(torch.int32))
        #     print((F.softmax(self.mix_logits.view(-1)) * batch_size).to(torch.int32))

        #     print("Dead Mask")
        #     print(is_dead_centers)
        #     # print("Drop Out")
        #     # print(drop_out_mask)

        #     max_post_index = torch.argmax(posterior, dim = 1)[:5]
        #     print("log mix probs")
        #     print(log_mix_probs[max_post_index])
        #     print("component_logits")
        #     print("Max")
        #     print(component_logits[torch.arange(5,dtype=torch.long),max_post_index])
        #     print(component_logits.mean(-1)[:5])
        #     print("Mdist")
        #     print(mahalanobis_dist.sum(-1)[torch.arange(5,dtype=torch.long),max_post_index])
        #     print(mahalanobis_dist.sum(-1).mean(-1)[:5])

    # def compute_logits(self, input: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    #     """
    #     Args:
    #         input: [B, D] where B is batch size, D is model dimension
    #     Returns:
    #         expert_logits: [B, E] logits for each expert
    #         gmm_logits: [B, E*C] logits for each component
    #     """

    #     B, C, D= input.shape
    #     input = input.reshape(-1,D)
    #     projected_input, recover_loss = self.projection(input.detach())

    #     input = input.float()
    #     projected_input = projected_input.detach().float()

    #     batch_size = projected_input.size(0)
        
    #     diff = (projected_input.unsqueeze(1) - self.means.unsqueeze(0))

    #     # [E*C, D] -> exponential -> [E*C, D]
    #     # vars = torch.exp(self.log_vars)
    #     vars = self.soft_plus(self.vars)  # [E*C, D]
    #     log_vars = torch.log(vars) # [E*C, D]
    #     # Sum over D dimension: [E*C, D] -> [E*C]
    #     log_det_term = torch.sum(log_vars, dim=-1)

    #     # [B, E*C, D] * [B, E*C, D] / [1, 1, D] -> sum over D -> [B, E*C]
    #     mahalanobis_dist = diff * diff / (vars + 1e-6)

    #     # Gaussian log-likelihood for each component
    #     # [B, E*C]
    #     component_logits = -0.5 * (
    #         # mahalanobis_dist +  # [B, E*C]
    #         torch.sum(mahalanobis_dist, dim=-1) + 
    #         log_det_term +      # [E*C]
    #         self.projection_dim * math.log(2 * math.pi)  # scalar
    #     )
        
    #     # mix_probs = F.softmax(self.mix_logits.view(-1),dim=0)
    #     log_mix_probs = F.log_softmax(self.mix_logits.view(-1), dim=0)  # [E*C]
    #     gmm_logits = component_logits + log_mix_probs.unsqueeze(0)  # [B, E*C]


    #     dead_logits = None
    #     if self.training:
    #         is_dead_centers = self.is_dead_center()
    #         if sum(is_dead_centers.int()) >= 2:
    #             dead_log_vars = log_vars[is_dead_centers,:]
    #             dead_det_term = torch.sum(dead_log_vars,dim=-1)
    #             dead_mdist = mahalanobis_dist[:,is_dead_centers,:]
    #             dead_component_logits = -0.5 * (
    #                 dead_mdist.sum(dim=-1) +
    #                 dead_det_term + 
    #                 self.projection_dim * math.log(2 * math.pi)
    #             )
    #             dead_log_mix_probs = F.log_softmax(self.mix_logits[is_dead_centers])
    #             dead_logits = dead_component_logits + dead_log_mix_probs.unsqueeze(0)
        

    #     # drop_out_mask = self.adaptive_dropout()
    #     # gmm_logits = gmm_logits.masked_fill(drop_out_mask,float("-inf"))
                
    #     if self.should_print():
    #         print("$"*100)
    #         print("Vars mean")
    #         print(vars.mean(dim=-1))
    #         print("logs Vars Sum")
    #         print(log_det_term)
    #         print("Diff")
    #         print((diff*diff).sum(-1)[:5,:])
    #         print("MDist")
    #         print(mahalanobis_dist.sum(dim=-1)[:5,:])

    #         def inspect_means_dist():
    #             with torch.no_grad():
    #                 mean_diff = self.means.unsqueeze(dim=0) - self.means.unsqueeze(dim=1)
    #                 mean_mdist = (mean_diff * mean_diff).sum(dim=-1)
    #                 print("Distance between centers")
    #                 print(mean_mdist)

    #         inspect_means_dist()

    #         print("Posterior")
    #         posterior = F.softmax(gmm_logits, dim = -1)
    #         print(posterior[:5,:])
    #         print(posterior.sum(dim=0).to(torch.int32))
    #         print((F.softmax(self.mix_logits.view(-1)) * batch_size).to(torch.int32))

    #         print("Dead Mask")
    #         print(is_dead_centers)
    #         # print("Drop Out")
    #         # print(drop_out_mask)

    #         max_post_index = torch.argmax(posterior, dim = 1)[:5]
    #         print("log mix probs")
    #         print(log_mix_probs[max_post_index])
    #         print("component_logits")
    #         print("Max")
    #         print(component_logits[torch.arange(5,dtype=torch.long),max_post_index])
    #         print(component_logits.mean(-1)[:5])
    #         print("Mdist")
    #         print(mahalanobis_dist.sum(-1)[torch.arange(5,dtype=torch.long),max_post_index])
    #         print(mahalanobis_dist.sum(-1).mean(-1)[:5])

    #     return gmm_logits, dead_logits ,recover_loss


class xGate(TopKGate):
    """Gate module which implements Top2Gating as described in Gshard_ with "Sharpen Aux Loss".
    ::

        gate = TopKGate(model_dim, num_experts)
        l_aux, combine_weights, dispatch_mask = gate(input)

    .. Gshard_: https://arxiv.org/pdf/2006.16668.pdf

    Args:
        model_dim (int):
            size of model embedding dimension
        num_experts (int):
            number of experts in model
    """

    wg: torch.nn.Linear

    def __init__(
        self,
        model_dim: int,
        num_experts: int,
        k: int = 1,
        projection_dim: int= 16,
        capacity_factor: float = 1.0,
        eval_capacity_factor: float = 1.0,
        min_capacity: int = 8,
        noisy_gate_policy: Optional[str] = None,
        drop_tokens: bool = True,
        use_rts: bool = True,
        ep_group: Union[torch.distributed.ProcessGroup, None] = None,
        top2_2nd_expert_sampling: bool = True,
        l_aux_type: str = "load_balancing",
        ) -> None:
        super().__init__(model_dim,num_experts,k,capacity_factor,eval_capacity_factor,min_capacity,noisy_gate_policy,drop_tokens,use_rts,ep_group,top2_2nd_expert_sampling)
        self.wg = nn.Linear(
            in_features = projection_dim,
            out_features = num_experts,
            bias = False
        )
        self.step_counter = 0
        self.debugging = True
        self.print_interval = 50
        self.projection = nn.Linear(
            in_features = model_dim,
            out_features = projection_dim,
            bias = False
        )
        self.l_aux_type = l_aux_type

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

    def forward(
        self,
        input: torch.Tensor,
        used_token: torch.Tensor = None,
        use_tutel: bool = False
        ) -> Tuple[Tensor, Tensor, Tensor]:  # type: ignore

        if self.wall_clock_breakdown:
            self.timers(TOPK_GATE_TIMER).start()

        input_fp32 = input.float()

        # input jittering
        if self.noisy_gate_policy == 'Jitter' and self.training:
            input_fp32 = multiplicative_jitter(input_fp32, device=input.device)

        routing_feature = self.projection(input_fp32)
        norm_logits = torch.nn.functional.linear(F.normalize(routing_feature,p=2,dim=1),weight=F.normalize(self.wg.weight,p=2,dim=1))

        if self.should_print():
            with torch.no_grad():
                # print("Logits")
                # print(expert_logits[:5,:])
                print(norm_logits[:5,:])
                print(norm_logits[-5:,:])

        self.step_counter += 1

        if self.k == 1:
            gate_output = top1gating(norm_logits, self.capacity_factor if self.training else self.eval_capacity_factor,
                                     self.min_capacity, used_token, self.noisy_gate_policy if self.training else None,
                                     self.drop_tokens, self.use_rts, self.l_aux_type,self.ep_group, use_tutel)

        else:
            gate_output = top2gating(norm_logits, self.capacity_factor if self.training else self.eval_capacity_factor,
                                     self.min_capacity, self.drop_tokens, self.l_aux_type, self.ep_group, self.top2_2nd_expert_sampling)

        if self.wall_clock_breakdown:
            self.timers(TOPK_GATE_TIMER).stop()
            self.gate_time = self.timers(TOPK_GATE_TIMER).elapsed(reset=False)

        return gate_output



class AuxFreeGate(TopKGate):
    """Gate module which implements Top2Gating as described in Gshard_ with "Sharpen Aux Loss".
    ::

        gate = TopKGate(model_dim, num_experts)
        l_aux, combine_weights, dispatch_mask = gate(input)

    .. Gshard_: https://arxiv.org/pdf/2006.16668.pdf

    Args:
        model_dim (int):
            size of model embedding dimension
        num_experts (int):
            number of experts in model
    """

    wg: torch.nn.Linear

    def __init__(
        self,
        model_dim: int,
        num_experts: int,
        k: int = 1,
        capacity_factor: float = 1.0,
        eval_capacity_factor: float = 1.0,
        min_capacity: int = 8,
        noisy_gate_policy: Optional[str] = None,
        drop_tokens: bool = True,
        use_rts: bool = True,
        ep_group: Union[torch.distributed.ProcessGroup, None] = None,
        top2_2nd_expert_sampling: bool = True,
        l_aux_type: str = "load_balancing",
        ) -> None:
        super().__init__(model_dim,num_experts,k,capacity_factor,eval_capacity_factor,min_capacity,noisy_gate_policy,drop_tokens,use_rts,ep_group,top2_2nd_expert_sampling)
        self.expert_BIAS = nn.Parameter(
            torch.zeros(num_experts),
            requires_grad=True # No gradient needed since we manually update
        )
        self.step_counter = 0
        self.debugging = True
        self.print_interval = 50
        self.l_aux_type = l_aux_type

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

    def forward(
        self,
        input: torch.Tensor,
        used_token: torch.Tensor = None,
        use_tutel: bool = False
        ) -> Tuple[Tensor, Tensor, Tensor]:  # type: ignore

        if self.wall_clock_breakdown:
            self.timers(TOPK_GATE_TIMER).start()

        input_fp32 = input.float()

        # input jittering
        if self.noisy_gate_policy == 'Jitter' and self.training:
            input_fp32 = multiplicative_jitter(input_fp32, device=input.device)

        logits = torch.nn.functional.linear(input_fp32, weight=self.wg.weight.float(), bias=None)
        logits = logits + self.expert_BIAS


        if self.k == 1:
            gate_output = top1gating(logits, self.capacity_factor if self.training else self.eval_capacity_factor,
                                     self.min_capacity, used_token, self.noisy_gate_policy if self.training else None,
                                     self.drop_tokens, self.use_rts, self.l_aux_type,self.ep_group, use_tutel)

        else:

            gate_output = list(top2gating(logits,
                                   self.capacity_factor if self.training else self.eval_capacity_factor,
                                   self.min_capacity, self.drop_tokens, self.l_aux_type, self.ep_group,
                                   self.top2_2nd_expert_sampling))
            expert_counts = gate_output[3]
            expert_counts_float = expert_counts.float().to(logits.device)
            gate_output[0] = (self.expert_BIAS * torch.sign(expert_counts_float - expert_counts_float.mean())).abs().sum()
            gate_output = tuple(gate_output)

        if self.should_print():
            with torch.no_grad():
                # print("Logits")
                # print(expert_logits[:5,:])
                print(f"Expert Count")
                print(gate_output[3])
                print(f"Loss:")
                print(gate_output[0])
                print(logits[:5,:])
                print(logits[-5:,:])

        self.step_counter += 1

        if self.wall_clock_breakdown:
            self.timers(TOPK_GATE_TIMER).stop()
            self.gate_time = self.timers(TOPK_GATE_TIMER).elapsed(reset=False)

        return gate_output