from typing import Callable, Dict, TYPE_CHECKING, Any, Optional, Tuple, Union
from deepspeed.utils.timer import SynchronizedWallClockTimer
from deepspeed.utils import logger
from deepspeed.utils.bwc import bwc_tensor_model_parallel_world_size
from deepspeed import comm as dist
from deepspeed.utils import groups
from deepspeed.moe.layer import MoE
from deepspeed.moe.sharded_moe import TopKGate
from deepspeed.moe.sharded_moe import MOELayer
from deepspeed.moe.sharded_moe import gumbel_rsample, _top_idx, _capacity, _one_hot_to_float, _AllToAll, einsum, MOE_TIMER, FIRST_ALLTOALL_TIMER, SECOND_ALLTOALL_TIMER
from deepspeed.moe.mappings import drop_tokens, gather_tokens

import torch.nn as nn
import torch
from torch import Tensor
from torch.nn import Module
import torch.nn.functional as F

from .Gaussian_Gating import GMMGate as GMMGateV2
from .sharpen_gating import TopKSharpenGate, GMMGate, xGate, AuxFreeGate

try:
    # To enable Tutel MoE optimizations:
    #   python3 -m pip install --user --upgrade git+https://github.com/microsoft/tutel@v0.1.x
    from tutel import moe as tutel_moe
    TUTEL_INSTALLED = True
except:
    # Fail silently so we don't spam logs unnecessarily if user isn't using tutel
    TUTEL_INSTALLED = False
    pass

def MOELayer_forward(self):
    def forward(*input: Tensor, **kwargs: Any) -> Tensor:

        if self.wall_clock_breakdown:
            self.timers(MOE_TIMER).start()

        # Implement Algorithm 2 from GShard paper.
        d_model = input[0].shape[-1]

        # Initial implementation -> Reshape into S tokens by dropping sequence dimension.
        # Reshape into G groups so that each group can distribute tokens equally
        # group_size = kwargs['group_size'] if 'group_size' in kwargs.keys() else 1
        B, C, D = input[0].shape
        reshaped_input = input[0].reshape(-1, d_model)

        if self.use_tutel:
            self.l_aux, C, E, indices_, locations_, gates_, self.exp_counts = self.gate(reshaped_input, input[1], True)
            S, M = reshaped_input.size(0), reshaped_input.size(1)

            if not hasattr(self, '_tutel_dispatcher'):
                self._tutel_dispatcher = tutel_moe.fast_dispatcher(E, C, M, dispatch_dtype=reshaped_input.dtype)
            self._tutel_dispatcher.update(indices_, locations_, gates_, capacity=C)
            dispatched_input = self._tutel_dispatcher.encode(reshaped_input)
        else:
            # self.l_aux, combine_weights, dispatch_mask, self.exp_counts = self.gate(reshaped_input, input[1])
            self.l_aux, combine_weights, dispatch_mask, self.exp_counts, self.gating_logits = self.gate(reshaped_input, input[1])
            self.gating_logits = self.gating_logits.view(B,C,-1)
            dispatched_input = einsum("sec,sm->ecm", dispatch_mask.type_as(input[0]), reshaped_input)

        if self.wall_clock_breakdown:
            self.timers(FIRST_ALLTOALL_TIMER).start()

        tensor_model_world_size = bwc_tensor_model_parallel_world_size(groups.mpu)
        if tensor_model_world_size > 1:
            # If the non-expert is tensor-parallel,
            # Whether expert is tensor-parallel or not , it will create
            # duplicate tokens on the tensor-parallel ranks.
            # drop duplicate tokens also doubles up as a communication
            # optimization as we are reducing the all-to-all communication volume.
            # 1: for not tensor-parallel expert,drop duplicate tokens to ensure
            # both correctness and reduce all-to-all communication.
            # 2: for tensor-parallel expert,drop duplicate tokens to reduce all-to-all
            # communication volume,before expert execution, it is necessary to perform
            # an allgather to ensure correctness,
            dispatched_input = drop_tokens(dispatched_input, dim=1)


        # rank = torch.distributed.get_rank()
        # world_size = torch.distributed.get_world_size()
        # # Get EP group members (this syntax depends on how EP groups are defined in your code)
        # ep_group_size = torch.distributed.get_world_size(group=self.ep_group)
        # ep_group_rank = torch.distributed.get_rank(group=self.ep_group)
        
        # print(f"Rank {rank}/{world_size} (EP rank {ep_group_rank}/{ep_group_size}): "
        #     f"Before drop_tokens, tensor shape={dispatched_input.shape}, "
        #     f"numel={dispatched_input.numel()}")

        dispatched_input = _AllToAll.apply(self.ep_group, dispatched_input)

        if self.wall_clock_breakdown:
            self.timers(FIRST_ALLTOALL_TIMER).stop()
            self.time_falltoall = self.timers(FIRST_ALLTOALL_TIMER).elapsed(reset=False)

        if tensor_model_world_size > 1 and groups._get_expert_model_parallel_world_size() > 1:
            # if both expert and non-expert are tensor-parallel
            # the dropped duplicate tokens need to be gathered on each
            # tensor parallel rank again to ensure correctness
            dispatched_input = gather_tokens(dispatched_input, dim=1)

        # Re-shape after all-to-all: ecm -> gecm
        dispatched_input = dispatched_input.reshape(self.ep_size, self.num_local_experts, -1, d_model)
        expert_output = self.experts(dispatched_input)
        # Re-shape before drop_tokens: gecm -> ecm
        expert_output = expert_output.reshape(self.ep_size * self.num_local_experts, -1, d_model)
        if tensor_model_world_size > 1 and groups._get_expert_model_parallel_world_size() > 1:
            # if both expert and non-expert are tensor-parallel
            # drop duplicate tokens to ensure both correctness
            # and reduce all-to-all communication.
            expert_output = drop_tokens(expert_output, dim=1)

        if self.wall_clock_breakdown:
            self.timers(SECOND_ALLTOALL_TIMER).start()

        expert_output = _AllToAll.apply(self.ep_group, expert_output)

        if self.wall_clock_breakdown:
            self.timers(SECOND_ALLTOALL_TIMER).stop()
            self.time_salltoall = self.timers(SECOND_ALLTOALL_TIMER).elapsed(reset=False)

        if tensor_model_world_size > 1:
            # the dropped duplicate tokens need to be gathered on each
            # tensor parallel rank again for the tensor-parallel
            # non-expert of the next layer.
            expert_output = gather_tokens(expert_output, dim=1)

        if self.use_tutel:
            combined_output = self._tutel_dispatcher.decode(expert_output.view(E * C, M))
        else:
            combined_output = einsum("sec,ecm->sm", combine_weights.type_as(input[0]), expert_output)

        a = combined_output.reshape(input[0].shape)

        if self.wall_clock_breakdown:
            self.timers(MOE_TIMER).stop()
            self.time_moe = self.timers(MOE_TIMER).elapsed(reset=False)

        return a
    return forward

class SharpenMoE(MoE):
    def __init__(
        self,
        hidden_size: int,
        expert: nn.Module,
        num_experts: int = 1,
        ep_size: int = 1,
        k: int = 1,
        components_per_expert=8,
        capacity_factor: float = 1.0,
        eval_capacity_factor: float = 1.0,
        min_capacity: int = 4,
        use_residual: bool = False,
        noisy_gate_policy: Optional[str] = None,
        drop_tokens: bool = True,
        use_rts: bool = True,
        use_tutel: bool = False,
        enable_expert_tensor_parallelism: bool = False,
        top2_2nd_expert_sampling: bool = True,
        l_aux_type: str = "load_balancing",
        routing_dim = 32,
        group_reactivation: bool = True,
        ) -> None:

        super().__init__(
        hidden_size,
        expert,
        num_experts,
        ep_size,
        k,
        capacity_factor,
        eval_capacity_factor,
        min_capacity,
        use_residual,
        noisy_gate_policy,
        drop_tokens,
        use_rts,
        use_tutel,
        enable_expert_tensor_parallelism,
        top2_2nd_expert_sampling,
        )
        if l_aux_type == "gaussian":
            self.deepspeed_moe.gate = GMMGate(
                model_dim=hidden_size, 
                num_experts=num_experts, 
                components_per_expert=components_per_expert,
                k = k, 
                projection_dim= routing_dim,
                capacity_factor= capacity_factor, 
                eval_capacity_factor= eval_capacity_factor,
                min_capacity=min_capacity, 
                drop_tokens=drop_tokens, 
                use_rts=use_rts, 
                ep_group=None,
                top2_2nd_expert_sampling=top2_2nd_expert_sampling,
                group_reactivation= group_reactivation,
            )
        elif l_aux_type == "xMoE":
            self.deepspeed_moe.gate = xGate(
                model_dim=hidden_size, 
                num_experts=num_experts, 
                k = k, 
                projection_dim= routing_dim,
                capacity_factor= capacity_factor, 
                eval_capacity_factor= eval_capacity_factor,
                min_capacity=min_capacity, 
                drop_tokens=drop_tokens, 
                use_rts=use_rts, 
                ep_group=None,
                top2_2nd_expert_sampling=top2_2nd_expert_sampling,
            )
        elif l_aux_type == "aux_free":
            self.deepspeed_moe.gate = AuxFreeGate(
                model_dim= hidden_size,
                num_experts=num_experts,
                k=k,
                capacity_factor=capacity_factor,
                eval_capacity_factor=eval_capacity_factor,
                min_capacity=min_capacity,
                noisy_gate_policy=noisy_gate_policy,
                drop_tokens=drop_tokens,
                use_rts=use_rts,
                ep_group=None,
                top2_2nd_expert_sampling=top2_2nd_expert_sampling,
                l_aux_type=l_aux_type,
            )
        else:
            self.deepspeed_moe.gate = TopKSharpenGate(
                model_dim= hidden_size,
                num_experts=num_experts,
                k=k,
                capacity_factor=capacity_factor,
                eval_capacity_factor=eval_capacity_factor,
                min_capacity=min_capacity,
                noisy_gate_policy=noisy_gate_policy,
                drop_tokens=drop_tokens,
                use_rts=use_rts,
                ep_group=None,
                top2_2nd_expert_sampling=top2_2nd_expert_sampling,
                l_aux_type=l_aux_type,
            )
        self.deepspeed_moe.forward = MOELayer_forward(self.deepspeed_moe)

    def forward(self,
                hidden_states: torch.Tensor,
                used_token: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """ MoE forward

        Arguments:
            hidden_states (Tensor): input to the layer
            used_token (Tensor, optional): default: None, mask only used tokens

        Returns:
            A tuple including output, gate loss, and expert count.

            * output (Tensor): output of the model

            * l_aux (Tensor): gate loss value

            * exp_counts (Tensor): expert count
        """
        output = self.deepspeed_moe(hidden_states, used_token)
        if self.use_residual:
            # Residual MoE
            output_mlp = self.mlp(hidden_states)
            if isinstance(output_mlp, tuple):
                output_mlp = output_mlp[0]  # Ignore the bias term for now
            coef = self.coefficient(hidden_states)
            coef = F.softmax(coef, dim=-1)
            output = output * coef[..., 0:1] + output_mlp * coef[..., 1:]
        return output, self.deepspeed_moe.l_aux, self.deepspeed_moe.exp_counts, self.deepspeed_moe.gating_logits