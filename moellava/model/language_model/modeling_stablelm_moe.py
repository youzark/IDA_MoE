from typing import List, Optional, Tuple, Union, Dict
import math
import os

import torch
import torch.nn as nn

from transformers import AutoConfig, AutoModelForCausalLM
from .stablelm.configuration_stablelm_epoch import StableLMEpochConfig
from .stablelm.modeling_stablelm_epoch import StableLMEpochModel, StableLMEpochForCausalLM, DecoderLayer

from transformers.modeling_outputs import CausalLMOutputWithPast

from ..llava_arch import LlavaMetaModel, LlavaMetaForCausalLM

# from deepspeed.moe.layer import MoE
from moellava.model.moe.sharpen_moe import SharpenMoE
from dataclasses import dataclass
from typing import Optional, Tuple, Union, List
import torch.nn as nn
from torch.nn import functional as F
from einops import rearrange
from torch.nn import CrossEntropyLoss
from transformers.models.llama.modeling_llama import logger
from transformers.utils import ModelOutput

class MoELLaVAStablelmConfig(StableLMEpochConfig):
    model_type = "moe_llava_stablelm"
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
    group_reactivation: bool = True,
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
        super(MoELLaVAStablelmConfig, self).__init__(**kwargs)
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
            group_reactivation = group_reactivation,
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
        if model_args is None:
            return
        new_moe_config = dict(
            ep_size= getattr(model_args,"ep_size",self.moe["ep_size"]),
            top_k_experts=getattr(model_args,"top_k_experts",self.moe["top_k_experts"]),
            capacity_factor=getattr(model_args,"capacity_factor",self.moe["capacity_factor"]),
            eval_capacity_factor=getattr(model_args,"eval_capacity_factor",self.moe["eval_capacity_factor"]),
            min_capacity=getattr(model_args,"min_capacity",self.moe["min_capacity"]),
            use_residual=getattr(model_args,"use_residual",self.moe["use_residual"]),
            router_aux_loss_coef=getattr(model_args,"router_aux_loss_coef",self.moe["router_aux_loss_coef"]),
            train_modules=getattr(model_args,"train_modules",self.moe["train_modules"]),
            l_aux_type=getattr(model_args,"l_aux_type",getattr(self.moe,"l_aux_type","load_balancing")),
            group_reactivation=getattr(model_args,"group_reactivation",getattr(self.moe,"group_reactivation",True))
        )
        self.moe.update(new_moe_config)
        # LoRA config
        new_lora_config = dict(
            lora_alpha=getattr(model_args,"lora_alpha",self.lora["lora_alpha"]),
            lora_dropout=getattr(model_args,"lora_dropout",self.lora["lora_dropout"]),
        )
        self.lora.update(new_lora_config)

@dataclass
class MoEBaseModelOutputWithPast(ModelOutput):
    last_hidden_state: torch.FloatTensor = None
    past_key_values: Optional[Tuple[Tuple[torch.FloatTensor]]] = None
    hidden_states: Optional[Tuple[torch.FloatTensor]] = None
    attentions: Optional[Tuple[torch.FloatTensor]] = None
    moe_metrics_list: Optional[List[Dict]] = None
    moe_loss_list: Optional[Tuple[torch.FloatTensor]] = None


@dataclass
class MoECausalLMOutputWithPast(ModelOutput):
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

class StableLMMLPLoraPath(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.scaling = config.lora["lora_alpha"] / config.lora["lora_rank"]
        self.dropout = nn.Dropout(p=config.lora['lora_dropout'])
        
        # Determine bias usage
        use_bias = config.lora['lora_bias'] in ['all', 'lora_only']
        
        # LoRA layers for gate projection
        self.lora_gate_A = nn.Linear(config.hidden_size, config.lora["lora_rank"], bias=False)
        self.lora_gate_B = nn.Linear(config.lora["lora_rank"], config.intermediate_size, bias=use_bias)
        
        # LoRA layers for up projection
        self.lora_up_A = nn.Linear(config.hidden_size, config.lora["lora_rank"], bias=False)
        self.lora_up_B = nn.Linear(config.lora["lora_rank"], config.intermediate_size, bias=use_bias)
        
        # LoRA layers for down projection
        self.lora_down_A = nn.Linear(config.intermediate_size, config.lora["lora_rank"], bias=False)
        self.lora_down_B = nn.Linear(config.lora["lora_rank"], config.hidden_size, bias=use_bias)
        
        self.act_fn = nn.SiLU()
        self._init_weights()
    
    def _init_weights(self):
        # Initialize LoRA A matrices
        for module in [self.lora_gate_A, self.lora_up_A, self.lora_down_A]:
            nn.init.kaiming_uniform_(module.weight, a=math.sqrt(5))
        
        # Initialize LoRA B matrices to zero
        for module in [self.lora_gate_B, self.lora_up_B, self.lora_down_B]:
            nn.init.zeros_(module.weight)

    def forward(self, hidden_states):
        # LoRA paths with dropout for gate and up projections
        lora_gate = self.lora_gate_B(self.dropout(self.lora_gate_A(hidden_states))) * self.scaling
        lora_up = self.lora_up_B(self.dropout(self.lora_up_A(hidden_states))) * self.scaling
        
        # Combine and activate
        intermediate = self.act_fn(lora_gate) * lora_up
        
        # Down projection
        lora_output = self.lora_down_B(self.dropout(self.lora_down_A(intermediate))) * self.scaling
        
        return lora_output


class MoELoRAStablelmMLP(nn.Module):
    def __init__(self, config, base_mlp, num_experts):
        """
        Args:
            config: Configuration object with lora and moe settings
            base_mlp: Original StableLM MLP with pretrained weights
            num_experts: Number of expert models
        """
        super().__init__()
        self.base_mlp = base_mlp
        # Freeze base MLP parameters
        for param in self.base_mlp.parameters():
            param.requires_grad = False
            
        # Create MoE layer for LoRA paths
        lora_expert = StableLMMLPLoraPath(config)
        
        self.moe = SharpenMoE(
            config.hidden_size,
            expert=lora_expert,
            num_experts=num_experts,
            ep_size=config.moe['ep_size'],
            k=config.moe['top_k_experts'],
            capacity_factor=config.moe['capacity_factor'],
            eval_capacity_factor=config.moe['eval_capacity_factor'],
            min_capacity=config.moe['min_capacity'],
            use_residual=config.moe['use_residual'],
            l_aux_type=config.moe['l_aux_type'],
            group_reactivation=config.moe["group_reactivation"]
        )

    def forward(self, hidden_states, used_token=None):
        # Base path
        base_output = self.base_mlp(hidden_states)
        
        # MoE LoRA path
        moe_output, aux_loss, exp_counts = self.moe(hidden_states, used_token=used_token)
        
        # Combine outputs
        combined_output = base_output + moe_output
        
        return combined_output, aux_loss, exp_counts


class MoEDecoderLayer(DecoderLayer):
    def __init__(self, config, *args, **kwargs):
        super().__init__(config, *args, **kwargs)
        """
        Seperation of init_moe() from __init__ is to load dense models' checkpoint before expand the model to MoE structure.
        So if the ckpt is dense, init_moe should be called after loading ckpt,
        if the ckpt is moe, init_moe should be called before loading ckpt.
        """
        self.config = config
        self.is_moe_initialized = False

    def init_moe(self, config, num_experts):
        """Initialize MoE components for this block"""
        if self.is_moe_initialized:
            return
        moe_config = config.moe
            
        if config.lora.get('lora_enable', False):
            # Create MoE layer with LoRA experts
            original_mlp = self.mlp
            self.mlp = MoELoRAStablelmMLP(
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
                group_reactivation=getattr(config.moe,"group_reactivation",True), #config.moe["group_reactivation"]
            )

        self.is_moe_initialized = True

    def forward(
            self,
            hidden_states: torch.Tensor,
            attention_mask: Optional[torch.Tensor] = None,
            position_ids: Optional[torch.LongTensor] = None,
            past_key_value: Optional[Tuple[torch.Tensor]] = None,
            output_attentions: Optional[bool] = False,
            use_cache: Optional[bool] = False,
            # padding_mask: Optional[torch.LongTensor] = None,
    ) -> Tuple[torch.FloatTensor, Optional[Tuple[torch.FloatTensor, torch.FloatTensor]]]:
        residual = hidden_states

        hidden_states = self.input_layernorm(hidden_states)

        hidden_states, self_attn_weights, present_key_value = self.self_attn(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_value=past_key_value,
            output_attentions=output_attentions,
            use_cache=use_cache,
            # padding_mask=padding_mask,  # unuseful but conflict to flashattn
        )
        hidden_states = residual + hidden_states

        # Fully Connected
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)

        moe_losses = []
        moe_metrics = {}

        def compute_entropy(tensor):
            with torch.no_grad():
                entropy = tensor * torch.log(tensor)
                return -torch.sum(entropy, dim = -1)

        if isinstance(self.mlp, SharpenMoE) or isinstance(self.mlp,MoELoRAStablelmMLP):
            hidden_states, aux_loss, exp_counts, gating_logits  = hidden_states
            cv = compute_cv(torch.tensor(exp_counts))
            moe_metrics['cv'] = cv.item()
            moe_metrics['gating_Entropy'] = compute_entropy(gating_logits).mean(dim=-1)
            moe_metrics["gating_logits"] = gating_logits
            moe_metrics["exp_counts"] = exp_counts
            moe_losses.append(aux_loss)

        hidden_states = residual + hidden_states

        outputs = (hidden_states,)

        if output_attentions:
            outputs += (self_attn_weights,)

        if use_cache:
            outputs += (present_key_value,)

        outputs += (moe_metrics, moe_losses,)

        return outputs

class MoELLaVAStablelmModel(LlavaMetaModel, StableLMEpochModel):
    config_class = MoELLaVAStablelmConfig

    def __init__(self, config: StableLMEpochConfig):
        super(MoELLaVAStablelmModel, self).__init__(config)
        self.is_moe_initialized = False
        for i in range(config.num_hidden_layers):
            self.layers[i] = MoEDecoderLayer(config)

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
            self.layers[layer_idx].init_moe(config, num_experts)
            
        self.is_moe_initialized = True



    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        output_moe_loss: Optional[bool] = True,
    ) -> Union[Tuple, MoEBaseModelOutputWithPast]:
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        use_cache = use_cache if use_cache is not None else self.config.use_cache

        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # Retrieve input_ids and inputs_embeds
        if input_ids is not None and inputs_embeds is not None:
            raise ValueError(
                "You cannot specify both decoder_input_ids and decoder_inputs_embeds at the same time"
            )
        elif input_ids is not None:
            batch_size, seq_length = input_ids.shape
        elif inputs_embeds is not None:
            batch_size, seq_length, _ = inputs_embeds.shape
        else:
            raise ValueError(
                "You have to specify either decoder_input_ids or decoder_inputs_embeds"
            )

        seq_length_with_past = seq_length
        past_key_values_length = 0

        if position_ids is None:
            device = input_ids.device if input_ids is not None else inputs_embeds.device
            position_ids = torch.arange(
                past_key_values_length,
                seq_length + past_key_values_length,
                dtype=torch.long,
                device=device,
            )
            position_ids = position_ids.unsqueeze(0).view(-1, seq_length)
        else:
            position_ids = position_ids.view(-1, seq_length).long()

        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids)
        # Embed positions
        if self._use_flash_attention_2:
            # 2d mask is passed through the layers
            attention_mask = attention_mask if (attention_mask is not None and 0 in attention_mask) else None
        else:
            if attention_mask is None:
                attention_mask = torch.ones(
                    (batch_size, seq_length_with_past),
                    dtype=torch.bool,
                    device=inputs_embeds.device,
                )
            attention_mask = self._prepare_decoder_attention_mask(
                attention_mask,
                (batch_size, seq_length),
                inputs_embeds,
                past_key_values_length,
            )

        hidden_states = inputs_embeds

        if self.gradient_checkpointing and self.training:
            if use_cache:
                logger.warning_once(
                    "`use_cache=True` is incompatible with gradient checkpointing. Setting `use_cache=False`..."
                )
                use_cache = False

        # decoder layers
        all_hidden_states = () if output_hidden_states else None
        all_self_attns = () if output_attentions else None
        next_decoder_cache = () if use_cache else None
        all_moe_loss = [] if output_moe_loss else None
        all_moe_metrics = [] if output_moe_loss else None

        for idx, decoder_layer in enumerate(self.layers):
            if output_hidden_states:
                all_hidden_states += (hidden_states,)

            past_key_value = past_key_values[idx] if past_key_values is not None else None

            if self.gradient_checkpointing and self.training:

                def create_custom_forward(module):
                    def custom_forward(*inputs):
                        # None for past_key_value
                        return module(*inputs, past_key_value, output_attentions)

                    return custom_forward

                layer_outputs = torch.utils.checkpoint.checkpoint(
                    create_custom_forward(decoder_layer), hidden_states, attention_mask, position_ids, use_reentrant=False
                    # create_custom_forward(decoder_layer), hidden_states, attention_mask, position_ids
                )
            else:
                layer_outputs = decoder_layer(
                    hidden_states,
                    attention_mask=attention_mask,
                    position_ids=position_ids,
                    past_key_value=past_key_value,
                    output_attentions=output_attentions,
                    use_cache=use_cache
                )

            hidden_states = layer_outputs[0]

            if use_cache:
                next_decoder_cache += (layer_outputs[2 if output_attentions else 1],)

            if output_attentions:
                all_self_attns += (layer_outputs[1],)

            if output_moe_loss:
                all_moe_loss.extend(layer_outputs[-1])
                all_moe_metrics.append(layer_outputs[-2])

        hidden_states = self.norm(hidden_states)

        # add hidden states from the last decoder layer
        if output_hidden_states:
            all_hidden_states += (hidden_states,)

        next_cache = next_decoder_cache if use_cache else None
        if not return_dict:
            return tuple(
                v for v in [hidden_states, next_cache, all_hidden_states, all_self_attns, all_moe_loss] if
                v is not None)

        return MoEBaseModelOutputWithPast(
            last_hidden_state=hidden_states,
            past_key_values=next_cache,
            hidden_states=all_hidden_states,
            attentions=all_self_attns,
            moe_metrics_list=all_moe_metrics,
            moe_loss_list=all_moe_loss,
        )

class MoELLaVAStablelmForCausalLM(StableLMEpochForCausalLM, LlavaMetaForCausalLM):
    config_class = MoELLaVAStablelmConfig

    def __init__(self, config):
        super(StableLMEpochForCausalLM, self).__init__(config)
        self.model = MoELLaVAStablelmModel(config)
        self.vocab_size = config.vocab_size
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

        is_moe_config = hasattr(config, 'moe') and config.moe.get('moe_enable', False)
        if is_moe_config:
            self.model.init_moe(config)
            print("MoE has been initialized from the existing MoE configuration.")
        else:
            print("Loading MoE Model From Dense Model ...")
        # Initialize weights and apply final processing
        self.post_init()

    def get_model(self):
        return self.model

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
            assert isinstance(config, MoELLaVAStablelmConfig), f"Config Provided to MoELLaVAStablelmForCausalLM.from_pretrained_unified should be MoELLaVAStablelmConfig be get {type(config)}"

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
            config =  MoELLaVAStablelmConfig(
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
                group_reactivation= getattr(model_args, "group_reactivation", True),
                l_aux_type = model_args.l_aux_type,
                lora_enable= model_args.lora_enable,
                lora_rank = model_args.lora_r,
                lora_alpha= model_args.lora_alpha,
                lora_dropout = model_args.lora_dropout,    # Dropout probability for LoRA layers
                lora_bias = model_args.lora_bias,       # Bias training strategy: "none", "all", or "lora_only"
                **(config.to_dict()),
            )
            model.config = config ## substitute dense config with moe one
            model.model.config = config
            model.model.init_moe(config) ## expand dense structure to moe

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
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        images: Optional[torch.FloatTensor] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, MoECausalLMOutputWithPast]:

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

        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        hidden_states = outputs[0]
        logits = self.lm_head(hidden_states)
        logits = logits.float()

        loss = None
        ce_loss = None
        avg_ppl = None
        if labels is not None:
            # Shift so that tokens < n predict n
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()

            loss_fct_unreduce = CrossEntropyLoss(reduction="none")
            loss_unreduce = loss_fct_unreduce(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
            loss_pertoken = loss_unreduce.view_as(shift_labels)
            avg_ppl = torch.exp(torch.mean(loss_pertoken,dim=-1))

            # Flatten the tokens
            loss_fct = CrossEntropyLoss()
            shift_logits = shift_logits.view(-1, self.config.vocab_size)
            shift_labels = shift_labels.view(-1)
            # Enable model parallelism
            shift_labels = shift_labels.to(shift_logits.device)
            loss = loss_fct(shift_logits, shift_labels)
            ce_loss = loss.clone().detach()

        moe_loss, moe_losses = None, []
        if len(outputs[-1]) > 0:
            moe_loss_list = outputs[-1]
            for moe_loss in moe_loss_list:
                if moe_loss is not None:
                    moe_losses.append(moe_loss)
            moe_loss = self.config.moe["router_aux_loss_coef"] * sum(moe_losses)
            if labels is not None:
                loss += moe_loss

        # if not return_dict:
        #     output = (logits,) + outputs[1:]
        #     output = (moe_loss,) + output if moe_loss is not None else output
        #     return (loss,) + output if loss is not None else output

        return MoECausalLMOutputWithPast(
            loss=loss,
            moe_loss=moe_loss,
            ce_loss=ce_loss,
            avg_ppl=avg_ppl,
            logits=logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
            moe_metrics_list=outputs.moe_metrics_list,
            moe_loss_list=outputs.moe_loss_list,
        )

    def prepare_inputs_for_generation(self, input_ids, past_key_values=None, inputs_embeds=None, **kwargs):
        images = kwargs.pop("images", None)
        _inputs = super().prepare_inputs_for_generation(
            input_ids, past_key_values=past_key_values, inputs_embeds=inputs_embeds, **kwargs
        )
        if images is not None:
            _inputs['images'] = images
        return _inputs

AutoConfig.register("moe_llava_stablelm", MoELLaVAStablelmConfig)
AutoModelForCausalLM.register(MoELLaVAStablelmConfig, MoELLaVAStablelmForCausalLM)