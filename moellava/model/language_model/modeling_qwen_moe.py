from dataclasses import dataclass
from typing import List, Optional, Tuple, Union,Dict
import math
import os

import torch
from torch import nn
from torch.nn import CrossEntropyLoss
import torch.nn.functional as F

from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer
from .qwen.modeling_qwen import QWenLMHeadModel, QWenModel, QWenBlock, QWenAttention \
    , logger
from .qwen.configuration_qwen import QWenConfig


from moellava.model.moe.sharpen_moe import SharpenMoE
from transformers.modeling_outputs import CausalLMOutputWithPast, BaseModelOutputWithPast
from ..llava_arch import LlavaMetaModel, LlavaQWenMetaForCausalLM


local_rank = None


class MoELLaVAQWenConfig(QWenConfig):
    model_type = "moe_llava_qwen"
    def __init__(
    self,
    num_experts:List[int]=[4],
    moe_enable=True,
    moe_mode='sparse',
    moe_layers_idx=None,
    train_modules=[],
    ep_size=1,
    top_k_experts=2,
    capacity_factor=1.,
    eval_capacity_factor=1.,
    min_capacity=4,
    use_residual=False,
    router_aux_loss_coef=0.01,
    l_aux_type: str = "load_balancing",
    components_per_expert=8,
    routing_dim = 32,
    #Lora Related
    lora_enable=False,
    lora_rank=8,
    lora_alpha=32,
    lora_dropout: float = 0.05,    # Dropout probability for LoRA layers
    lora_bias: str = "none",       # Bias training strategy: "none", "all", or "lora_only"
    moe: Dict = None,
    lora: Dict = None,
    **kwargs
        ):
        super(MoELLaVAQWenConfig, self).__init__(**kwargs)
        num_layers = self.num_hidden_layers
        if moe_layers_idx is not None:
            moe_mode = 'custom'
            assert len(moe_layers_idx) <= num_layers
            assert max(moe_layers_idx) < num_layers
            assert min(moe_layers_idx) >= 0
        else:
            if moe_mode == "first_half":
                moe_layers_idx = list(range(0, num_layers // 2))
            elif moe_mode == "second_half":
                moe_layers_idx = list(range(num_layers // 2, num_layers))
            elif moe_mode == "sparse":
                moe_layers_idx = list(range(num_layers))[::2]
            elif moe_mode == "dense":
                moe_layers_idx = list(range(num_layers))
            else:
                raise NotImplementedError(
                    f'Only support ["first_half", "second_half", "sparse", "dense"], but found {moe_mode}')

        if len(num_experts) == 1:
            num_experts = num_experts * len(moe_layers_idx)
        assert len(num_experts) == len(moe_layers_idx), "the Customized num_experts doesn't match the length of moe_layers_idx"

        self.moe = dict(
            num_experts=num_experts,
            moe_enable=moe_enable,
            moe_mode=moe_mode,
            moe_layers_idx=moe_layers_idx,
            ep_size=ep_size,
            top_k_experts=top_k_experts,
            capacity_factor=capacity_factor,
            eval_capacity_factor=eval_capacity_factor,
            min_capacity=min_capacity,
            use_residual=use_residual,
            router_aux_loss_coef=router_aux_loss_coef,
            train_modules=train_modules,
            l_aux_type=l_aux_type,
            components_per_expert=components_per_expert,
            routing_dim = routing_dim,
        ) if moe is None else moe

        # LoRA config
        self.lora = dict(
            lora_enable=lora_enable,
            lora_rank=lora_rank,
            lora_alpha=lora_alpha,
            lora_dropout=lora_dropout,
            lora_bias= lora_bias,
        ) if lora is None else lora

        # Validate MoE and LoRA configuration
        if lora_enable and not moe_enable:
            raise ValueError("LoRA can only be enabled when MoE is enabled")

    def update_config(self, model_args):
        new_moe_config = dict(
            ep_size= getattr(model_args,"ep_size",self.moe["ep_size"]),
            top_k_experts=getattr(model_args,"top_k_experts",self.moe["top_k_experts"]),
            capacity_factor=getattr(model_args,"capacity_factor",self.moe["capacity_factor"]),
            eval_capacity_factor=getattr(model_args,"eval_capacity_factor",self.moe["eval_capacity_factor"]),
            min_capacity=getattr(model_args,"min_capacity",self.moe["min_capacity"]),
            use_residual=getattr(model_args,"use_residual",self.moe["use_residual"]),
            router_aux_loss_coef=getattr(model_args,"router_aux_loss_coef",self.moe["router_aux_loss_coef"]),
            train_modules=getattr(model_args,"train_modules",self.moe["train_modules"]),
            l_aux_type=getattr(model_args,"l_aux_type",self.moe["l_aux_type"]),
        )
        self.moe.update(new_moe_config)
        # LoRA config
        new_lora_config = dict(
            lora_alpha=getattr(model_args,"lora_alpha",self.lora["lora_alpha"]),
            lora_dropout=getattr(model_args,"lora_dropout",self.lora["lora_dropout"]),
        )
        self.lora.update(new_lora_config)

@dataclass
class MoEBaseModelOutputWithPast(BaseModelOutputWithPast):
    last_hidden_state: torch.FloatTensor = None
    past_key_values: Optional[Tuple[Tuple[torch.FloatTensor]]] = None
    hidden_states: Optional[Tuple[torch.FloatTensor]] = None
    attentions: Optional[Tuple[torch.FloatTensor]] = None
    moe_metrics_list: Optional[List[Dict]] = None
    moe_loss_list: Optional[Tuple[torch.FloatTensor]] = None

@dataclass
class MoECausalLMOutputWithPast(CausalLMOutputWithPast):
    loss: Optional[torch.FloatTensor] = None
    moe_loss: Optional[torch.FloatTensor] = None
    ce_loss: Optional[torch.FloatTensor] = None
    avg_ppl: Optional[torch.FloatTensor] = None
    logits: torch.FloatTensor = None
    past_key_values: Optional[Tuple[Tuple[torch.FloatTensor]]] = None
    hidden_states: Optional[Tuple[torch.FloatTensor]] = None
    attentions: Optional[Tuple[torch.FloatTensor]] = None
    moe_metrics_list: Optional[List[Dict]] = None
    moe_loss_list: Optional[Tuple[torch.FloatTensor]] = None

def compute_cv(expert_counts):
    # expert_counts: tensor of token counts per expert
    mean = torch.mean(expert_counts.float())
    std = torch.std(expert_counts.float())
    return std / (mean + 1e-8)

class QWenMLPLoraPath(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.scaling = config.lora["lora_alpha"] / config.lora["lora_rank"]
        self.dropout = nn.Dropout(p=config.lora['lora_dropout'])
        
        # Determine bias usage
        use_bias = config.lora['lora_bias'] in ['all', 'lora_only']
        
        # LoRA layers only
        self.lora_w1_A = nn.Linear(config.hidden_size,config.lora["lora_rank"], bias=False)
        self.lora_w1_B = nn.Linear(config.lora["lora_rank"], config.intermediate_size // 2, bias=use_bias)
        
        self.lora_w2_A = nn.Linear(config.hidden_size,config.lora["lora_rank"], bias=False)
        self.lora_w2_B = nn.Linear(config.lora["lora_rank"], config.intermediate_size // 2, bias=use_bias)
        
        self.lora_c_A = nn.Linear(config.intermediate_size // 2,config.lora["lora_rank"], bias=False)
        self.lora_c_B = nn.Linear(config.lora["lora_rank"], config.hidden_size, bias=use_bias)
        
        self._init_weights()
    
    def _init_weights(self):
        # Initialize LoRA A matrices
        for module in [self.lora_w1_A, self.lora_w2_A, self.lora_c_A]:
            nn.init.kaiming_uniform_(module.weight, a=math.sqrt(5))
        
        # Initialize LoRA B matrices to zero
        for module in [self.lora_w1_B, self.lora_w2_B, self.lora_c_B]:
            nn.init.zeros_(module.weight)

    def forward(self, hidden_states):
        # LoRA paths with dropout
        lora_a1 = self.lora_w1_B(self.dropout(self.lora_w1_A(hidden_states))) * self.scaling
        lora_a2 = self.lora_w2_B(self.dropout(self.lora_w2_A(hidden_states))) * self.scaling
        
        # Combine and activate
        intermediate_parallel = lora_a1 * F.silu(lora_a2)
        
        # Output projection
        lora_output = self.lora_c_B(self.dropout(self.lora_c_A(intermediate_parallel))) * self.scaling
        
        return lora_output

class MoELoRAQWenMLP(nn.Module):
    def __init__(self, config, base_mlp, num_experts):
        """
        Args:
            config: MoELLaVAQWenConfig
            base_mlp: Original QWenMLP with pretrained weights
        """
        super().__init__()
        self.base_mlp = base_mlp
        # Freeze base MLP parameters
        for param in self.base_mlp.parameters():
            param.requires_grad = False
            
        # Create MoE layer for LoRA paths
        lora_expert = QWenMLPLoraPath(config)
        
        self.moe = SharpenMoE(
            config.hidden_size,
            expert=lora_expert,
            num_experts=num_experts,
            ep_size=config.moe['ep_size'],
            k=config.moe['top_k_experts'],
            capacity_factor=config.moe['capacity_factor'],
            eval_capacity_factor=config.moe['eval_capacity_factor'],
            components_per_expert=config.moe["components_per_expert"],
            routing_dim = config.moe["routing_dim"],
            min_capacity=config.moe['min_capacity'],
            use_residual=config.moe['use_residual'],
            l_aux_type=config.moe['l_aux_type'],
        )

    def forward(self, hidden_states, used_token = None):
        # Base path
        base_output = self.base_mlp(hidden_states)
        
        # MoE LoRA path
        moe_output, aux_loss, exp_counts, gating_logits = self.moe(hidden_states,used_token = used_token)
        
        # Combine outputs
        combined_output = base_output + moe_output
        
        return combined_output, aux_loss, exp_counts, gating_logits

class MoEQWenBlock(QWenBlock):
    def __init__(self, config, *args, **kwargs):
        super().__init__(config, *args, **kwargs)
        """
        Seperation of init_moe() from __init__ is to load dense models' checkpoint before expand the model to MoE structure.
        So if the ckpt is dense, init_moe should be called after loading ckpt,
        if the ckpt is moe, init_moe should be called before loading ckpt.
        """

        self.is_moe_initialized = False

    def init_moe(self, config, num_experts):
        """Initialize MoE components for this block"""
        if self.is_moe_initialized:
            return
        moe_config = config.moe
            
        if config.lora.get('lora_enable', False):
            # Create MoE layer with LoRA experts
            original_mlp = self.mlp
            self.mlp = MoELoRAQWenMLP(
                config= config,
                base_mlp= original_mlp,
                num_experts=num_experts,
            )
        else:
            # Create regular MoE layer
            self.mlp = SharpenMoE(
                self.config.hidden_size,
                expert=self.mlp,
                num_experts=num_experts,
                ep_size=moe_config['ep_size'],
                k=moe_config['top_k_experts'],
                components_per_expert=moe_config["components_per_expert"],
                routing_dim= moe_config["routing_dim"],
                capacity_factor=moe_config['capacity_factor'],
                eval_capacity_factor=moe_config['eval_capacity_factor'],
                min_capacity=moe_config['min_capacity'],
                use_residual=moe_config['use_residual'],
                l_aux_type=moe_config.get('l_aux_type', "load_balancing"),
            )

        self.is_moe_initialized = True

    def forward(
        self,
        hidden_states: Optional[Tuple[torch.FloatTensor]],
        rotary_pos_emb_list: Optional[List[List[torch.Tensor]]] = None,
        layer_past: Optional[Tuple[torch.Tensor]] = None,
        attention_mask: Optional[torch.FloatTensor] = None,
        head_mask: Optional[torch.FloatTensor] = None,
        encoder_hidden_states: Optional[torch.Tensor] = None,
        encoder_attention_mask: Optional[torch.FloatTensor] = None,
        use_cache: Optional[bool] = False,
        output_attentions: Optional[bool] = False,
    ):
        layernorm_output = self.ln_1(hidden_states)

        attn_outputs = self.attn(
            layernorm_output,
            rotary_pos_emb_list,
            layer_past=layer_past,
            attention_mask=attention_mask,
            head_mask=head_mask,
            use_cache=use_cache,
            output_attentions=output_attentions,
        )
        attn_output = attn_outputs[0]
        outputs = attn_outputs[1:]

        residual = hidden_states
        layernorm_input = attn_output + residual
        layernorm_output = self.ln_2(layernorm_input)

        # Create used_token mask from attention_mask
        if attention_mask is not None:
            # attention_mask is typically [batch_size, 1, 1, seq_length] or [batch_size, seq_length]
            # We need [batch_size * seq_length] or just [num_tokens]
            used_token = (attention_mask > (torch.finfo(attention_mask.dtype).min / 2))
            used_token = used_token.squeeze().view(-1).float()
        else:
            used_token = None

        residual = layernorm_input
        mlp_output = self.mlp(layernorm_output, used_token = used_token) if isinstance(self.mlp,SharpenMoE) or isinstance(self.mlp,MoELoRAQWenMLP) else self.mlp(layernorm_output)
        
        moe_losses = []
        moe_metrics = {}

        def compute_entropy(tensor):
            with torch.no_grad():
                entropy = tensor * torch.log2(tensor)
                return -torch.sum(entropy, dim = -1)
        
        if isinstance(self.mlp, SharpenMoE) or isinstance(self.mlp,MoELoRAQWenMLP):
            mlp_output, aux_loss, exp_counts, gating_logits = mlp_output
            cv = compute_cv(torch.tensor(exp_counts))
            # print(f"exp_counts: {exp_counts},\ncv: {cv}")
            moe_metrics['cv'] = cv.item()
            moe_metrics['gating_Entropy'] = compute_entropy(gating_logits).mean(dim=-1)
            moe_losses.append(aux_loss)
            
        hidden_states = residual + mlp_output

        if use_cache:
            outputs = (hidden_states,) + outputs
        else:
            outputs = (hidden_states,) + outputs[1:]

        outputs += (moe_metrics, moe_losses,)
        return outputs


class MoELLaVAQWenModel(LlavaMetaModel,QWenModel):
    config_class = MoELLaVAQWenConfig

    def __init__(self, config):
        super().__init__(config)
        self.is_moe_initialized = False
        for i in range(config.num_hidden_layers):
            self.h[i] = MoEQWenBlock(config)

    def embed_tokens(self, input_ids):
        return self.wte(input_ids)

    def init_moe(self, config):
        """Initialize MoE components for specified layers"""
        if self.is_moe_initialized:
            return

        moe_layers_idx = config.moe['moe_layers_idx']
        num_experts_list = config.moe['num_experts']
        if len(num_experts_list) == 1:
            num_experts_list = num_experts_list * len(moe_layers_idx)
        assert len(num_experts_list) == len(moe_layers_idx)
        self.config.moe["num_experts"] = num_experts_list
        
        for layer_idx, num_experts in zip(moe_layers_idx, num_experts_list):
            self.h[layer_idx].init_moe(config, num_experts)
            
        self.is_moe_initialized = True


    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Tuple[Tuple[torch.Tensor]]] = None,
        attention_mask: Optional[torch.FloatTensor] = None,
        token_type_ids: Optional[torch.LongTensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        head_mask: Optional[torch.FloatTensor] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        encoder_hidden_states: Optional[torch.Tensor] = None,
        encoder_attention_mask: Optional[torch.FloatTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        output_moe_loss: Optional[bool] = True,
    ):
        output_attentions = (
            output_attentions
            if output_attentions is not None
            else self.config.output_attentions
        )
        output_hidden_states = (
            output_hidden_states
            if output_hidden_states is not None
            else self.config.output_hidden_states
        )
        use_cache = use_cache if use_cache is not None else self.config.use_cache
        return_dict = (
            return_dict if return_dict is not None else self.config.use_return_dict
        )

        if input_ids is not None and inputs_embeds is not None:
            raise ValueError(
                "You cannot specify both input_ids and inputs_embeds at the same time"
            )
        elif input_ids is not None:
            input_shape = input_ids.size()
            input_ids = input_ids.view(-1, input_shape[-1])
            batch_size = input_ids.shape[0]
        elif inputs_embeds is not None:
            input_shape = inputs_embeds.size()[:-1]
            batch_size = inputs_embeds.shape[0]
        else:
            raise ValueError("You have to specify either input_ids or inputs_embeds")

        device = input_ids.device if input_ids is not None else inputs_embeds.device

        if token_type_ids is not None:
            token_type_ids = token_type_ids.view(-1, input_shape[-1])
        if position_ids is not None:
            position_ids = position_ids.view(-1, input_shape[-1])

        if past_key_values is None:
            past_length = 0
            past_key_values = tuple([None] * len(self.h))
        else:
            if self.use_cache_quantization:
                past_length = past_key_values[0][0][0].size(2)
            else:
                past_length = past_key_values[0][0].size(-2)
        if position_ids is None:
            position_ids = torch.arange(
                past_length,
                input_shape[-1] + past_length,
                dtype=torch.long,
                device=device,
            )
            position_ids = position_ids.unsqueeze(0).view(-1, input_shape[-1])

        if attention_mask is not None:
            if batch_size <= 0:
                raise ValueError("batch_size has to be defined and > 0")
            attention_mask = attention_mask.view(batch_size, -1)
            attention_mask = attention_mask[:, None, None, :]
            attention_mask = attention_mask.to(dtype=self.dtype)
            attention_mask = (1.0 - attention_mask) * torch.finfo(self.dtype).min

        encoder_attention_mask = None
        head_mask = self.get_head_mask(head_mask, self.config.num_hidden_layers)

        if inputs_embeds is None:
            inputs_embeds = self.wte(input_ids)
        hidden_states = inputs_embeds

        kv_seq_len = hidden_states.size()[1]
        if past_key_values[0] is not None:
            # past key values[0][0] shape: bs * seq_len * head_num * dim
            if self.use_cache_quantization:
                kv_seq_len += past_key_values[0][0][0].shape[2]
            else:
                kv_seq_len += past_key_values[0][0].shape[1]

        if self.training or not self.use_dynamic_ntk:
            ntk_alpha_list = [1.0]
        elif kv_seq_len != hidden_states.size()[1]:
            ntk_alpha_list = self.rotary_emb._ntk_alpha_cached_list
        else:
            ntk_alpha_list = []
            if attention_mask is not None and kv_seq_len > self.seq_length:
                true_seq_lens = attention_mask.squeeze(1).squeeze(1).eq(0).sum(dim=-1, dtype=torch.int32)
                for i in range(hidden_states.size()[0]):
                    true_seq_len = true_seq_lens[i].item()
                    ntk_alpha = self.get_ntk_alpha(true_seq_len)
                    ntk_alpha_list.append(ntk_alpha)
            else:
                ntk_alpha = self.get_ntk_alpha(kv_seq_len)
                ntk_alpha_list.append(ntk_alpha)
        self.rotary_emb._ntk_alpha_cached_list = ntk_alpha_list
        rotary_pos_emb_list = [
            self.rotary_emb(kv_seq_len, ntk_alpha=ntk_alpha) for ntk_alpha in ntk_alpha_list
        ]

        hidden_states = self.drop(hidden_states)
        output_shape = input_shape + (hidden_states.size(-1),)

        if self.gradient_checkpointing and self.training:
            if use_cache:
                logger.warning_once(
                    "`use_cache=True` is incompatible with gradient checkpointing. Setting `use_cache=False`..."
                )
                use_cache = False

        presents = () if use_cache else None
        all_self_attentions = () if output_attentions else None
        all_hidden_states = () if output_hidden_states else None
        all_moe_loss = [] if output_moe_loss else None
        all_moe_metrics = [] if output_moe_loss else None

        for i, (block, layer_past) in enumerate(zip(self.h, past_key_values)):

            if output_hidden_states:
                all_hidden_states = all_hidden_states + (hidden_states,)

            if self.gradient_checkpointing and self.training:

                def create_custom_forward(module):
                    def custom_forward(*inputs):
                        # None for past_key_value
                        return module(*inputs, use_cache, output_attentions)

                    return custom_forward

                outputs = torch.utils.checkpoint.checkpoint(
                    create_custom_forward(block),
                    hidden_states,
                    rotary_pos_emb_list,
                    None,
                    attention_mask,
                    head_mask[i],
                    encoder_hidden_states,
                    encoder_attention_mask,
                    use_reentrant=False,
                )
            else:
                outputs = block(
                    hidden_states,
                    layer_past=layer_past,
                    rotary_pos_emb_list=rotary_pos_emb_list,
                    attention_mask=attention_mask,
                    head_mask=head_mask[i],
                    encoder_hidden_states=encoder_hidden_states,
                    encoder_attention_mask=encoder_attention_mask,
                    use_cache=use_cache,
                    output_attentions=output_attentions,
                )

            hidden_states = outputs[0]
            if use_cache is True:
                presents = presents + (outputs[1],)

            if output_attentions:
                all_self_attentions = all_self_attentions + (outputs[2 if use_cache else 1],)

            if output_moe_loss:
                all_moe_loss.extend(outputs[-1])
                all_moe_metrics.append(outputs[-2])

        hidden_states = self.ln_f(hidden_states)
        hidden_states = hidden_states.view(output_shape)
        # Add last hidden state
        if output_hidden_states:
            all_hidden_states = all_hidden_states + (hidden_states,)


        if not return_dict:
            return tuple(
                v for v in [hidden_states, presents, all_hidden_states, all_moe_loss] if v is not None
            )

        return MoEBaseModelOutputWithPast(
            last_hidden_state=hidden_states,
            past_key_values=presents,
            hidden_states=all_hidden_states,
            attentions=all_self_attentions,
            moe_metrics_list=all_moe_metrics,
            moe_loss_list=all_moe_loss,
        )

class MoELLaVAQWenForCausalLM(QWenLMHeadModel, LlavaQWenMetaForCausalLM):
    config_class = MoELLaVAQWenConfig

    def __init__(self, config):
        super().__init__(config)
        self.transformer = MoELLaVAQWenModel(config)
        # Check if loading an existing MoE model
        is_moe_config = hasattr(config, 'moe') and config.moe.get('moe_enable', False)
        if is_moe_config:
            self.transformer.init_moe(config)
            print("MoE has been initialized from the existing MoE configuration.")
        else:
            print("Loading MoE Model From Dense Model ...")
        self.post_init()

    @classmethod
    def from_pretrained_unified( 
        cls,
        *args,
        pretrained_model_name_or_path: Optional[Union[str, os.PathLike]],
        trust_remote_code=True,
        **kwargs,
        ):
        """
        Loading checkpoint from both Dense and Moe Version to Moe Model.
        """
        model_args = kwargs.pop('model_args', None)
        config = kwargs.pop("config", None)
        if config is not None:
            assert isinstance(config,  MoELLaVAQWenConfig), f"Config Provided to MoELLaVAQWenForCausalLM.from_pretrained_unified should be MoELLaVAQWenConfig be get {type(config)}"

        if config is None:
            config = AutoConfig.from_pretrained(pretrained_model_name_or_path, **kwargs)

        # Check if loading an existing MoE model
        is_moe_checkpoint = hasattr(config, 'moe') and config.moe.get('moe_enable', False)

        if not is_moe_checkpoint and model_args is None:
            raise ValueError("When loading MoE Model From a Dense Model, moe_config must be provided in from_pretrained() function")

        if is_moe_checkpoint:
            config.update_config(model_args)
            model = cls.from_pretrained(pretrained_model_name_or_path,*args,config=config,trust_remote_code=trust_remote_code,**kwargs)
        else:
            model = cls.from_pretrained(pretrained_model_name_or_path,*args,config=config,trust_remote_code=trust_remote_code,**kwargs) ## Load dense Model first
            config =  MoELLaVAQWenConfig(
                num_experts= model_args.num_experts,
                moe_enable = model_args.moe_enable,
                moe_mode = model_args.moe_mode,
                moe_layers_idx = model_args.moe_layers_idx,
                train_modules = model_args.train_modules,
                ep_size= model_args.ep_size,
                top_k_experts = model_args.top_k_experts,
                capacity_factor = model_args.capacity_factor,
                components_per_expert = getattr(model_args,"components_per_expert",1),
                routing_dim = getattr(model_args,"routing_dim",32),
                eval_capacity_factor = model_args.eval_capacity_factor,
                min_capacity = model_args.min_capacity,
                use_residual = model_args.use_residual,
                router_aux_loss_coef = model_args.router_aux_loss_coef,
                l_aux_type = model_args.l_aux_type,
                lora_enable= model_args.lora_enable,
                lora_rank = model_args.lora_r,
                lora_alpha= model_args.lora_alpha,
                lora_dropout = model_args.lora_dropout,    # Dropout probability for LoRA layers
                lora_bias = model_args.lora_bias,       # Bias training strategy: "none", "all", or "lora_only"
                **(config.to_dict()),
            )
            model.config = config ## substitute dense config with moe one
            model.transformer.config = config
            model.transformer.init_moe(config) ## expand dense structure to moe

        ## Set Trainable Parameters
        if config.moe['train_modules'] is not None and len(config.moe['train_modules']) > 0:
            for n, p in model.named_parameters():
                # For LoRA MoE version
                if config.lora.get('lora_enable', False):
                    for n, p in model.named_parameters():
                        # Train LoRA parameters that correspond to specified modules
                        if any(f"lora_{name}" in n for name in config.moe['train_modules']):
                            p.requires_grad = True
                        # Train MoE gate if 'wg' is in train_modules
                        elif 'wg' in config.moe['train_modules'] and "deepspeed_moe" in n:
                            p.requires_grad = True
                        # Freeze everything else
                        else:
                            p.requires_grad = False
                # For regular MoE version
                else:
                    if any(name in n and ("deepspeed_moe" in n or "coefficient" in n) for name in config.moe['train_modules']):
                        p.requires_grad = True
                    else:
                        p.requires_grad = False
        
        return model

    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Tuple[Tuple[torch.Tensor]]] = None,
        attention_mask: Optional[torch.FloatTensor] = None,
        token_type_ids: Optional[torch.LongTensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        head_mask: Optional[torch.FloatTensor] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        encoder_hidden_states: Optional[torch.Tensor] = None,
        encoder_attention_mask: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        images: Optional[torch.FloatTensor] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, MoECausalLMOutputWithPast]:
        # Handle multimodal inputs
        if inputs_embeds is None:
            (
                input_ids,
                position_ids,
                attention_mask,
                past_key_values,
                inputs_embeds,
                labels
            ) = self.prepare_inputs_labels_for_multimodal(
                input_ids,
                position_ids,
                attention_mask,
                past_key_values,
                labels,
                images
            )

        # Forward pass through transformer
        transformer_outputs = self.transformer(
            input_ids=input_ids,
            past_key_values=past_key_values,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            encoder_hidden_states=encoder_hidden_states,
            encoder_attention_mask=encoder_attention_mask,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        hidden_states = transformer_outputs[0]
        lm_logits = self.lm_head(hidden_states)

        loss = None
        ce_loss = None
        avg_ppl = None
        if labels is not None:
            labels = labels.to(lm_logits.device)
            shift_logits = lm_logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()

            loss_fct_unreduce = CrossEntropyLoss(reduction="none")
            loss_unreduce = loss_fct_unreduce(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
            loss_pertoken = loss_unreduce.view_as(shift_labels)
            # avg_ppl = torch.mean(loss_pertoken.exp(),dim=-1)
            avg_ppl = torch.exp(torch.mean(loss_pertoken,dim=-1))


            loss_fct = CrossEntropyLoss()
            loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
            ce_loss = loss.clone().detach()

        moe_loss, moe_losses = None, []
        if len(transformer_outputs[-1]) > 0:
            moe_loss_list = transformer_outputs[-1]
            for moe_loss in moe_loss_list:
                if moe_loss is not None:
                    moe_losses.append(moe_loss)
            moe_loss = self.config.moe["router_aux_loss_coef"] * sum(moe_losses)
            if labels is not None:
                loss += moe_loss

        # if not return_dict:
        #     output = (lm_logits,) + transformer_outputs[1:]
        #     output = (moe_loss,) + output if moe_loss is not None else output
        #     return ((loss,) + output) if loss is not None else output

        return MoECausalLMOutputWithPast(
            loss=loss,
            moe_loss=moe_loss,
            ce_loss=ce_loss,
            avg_ppl=avg_ppl,
            logits=lm_logits,
            past_key_values=transformer_outputs.past_key_values,
            hidden_states=transformer_outputs.hidden_states,
            attentions=transformer_outputs.attentions,
            moe_metrics_list=transformer_outputs.moe_metrics_list,
            moe_loss_list=transformer_outputs.moe_loss_list,
        )

    def prepare_inputs_for_generation(self, input_ids, past_key_values=None, inputs_embeds=None, **kwargs):
        images = kwargs.pop("images", None)
        _inputs = super().prepare_inputs_for_generation(
            input_ids, past_key_values=past_key_values, inputs_embeds=inputs_embeds, **kwargs
        )
        if images is not None:
            _inputs['images'] = images
        return _inputs

    def get_model(self):
        return self.transformer

AutoConfig.register("moe_llava_qwen", MoELLaVAQWenConfig)
AutoModelForCausalLM.register(MoELLaVAQWenConfig, MoELLaVAQWenForCausalLM)
