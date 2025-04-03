#    Copyright 2023 Haotian Liu
#
#    Licensed under the Apache License, Version 2.0 (the "License");
#    you may not use this file except in compliance with the License.
#    You may obtain a copy of the License at
#
#        http://www.apache.org/licenses/LICENSE-2.0
#
#    Unless required by applicable law or agreed to in writing, software
#    distributed under the License is distributed on an "AS IS" BASIS,
#    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#    See the License for the specific language governing permissions and
#    limitations under the License.


from typing import List, Optional, Tuple, Union

import torch
import torch.nn as nn

from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer
from .qwen2.modeling_qwen2 import Qwen2ForCausalLM, Qwen2Model
from .qwen2.configuration_qwen2 import Qwen2Config

from transformers.modeling_outputs import CausalLMOutputWithPast

from .qwen2.tokenization_qwen2 import  Qwen2Tokenizer
from ..llava_arch import LlavaMetaModel, LlavaMetaForCausalLM
import torch.distributed as dist

from moellava.utils import contain_nan

class LlavaQWen2Config(Qwen2Config):
    model_type = "llava_qwen2"
    def __init__(self, **kwargs):
        super(LlavaQWen2Config,self).__init__(**kwargs)

class LlavaQWen2Model(LlavaMetaModel, Qwen2Model):
    config_class = LlavaQWen2Config

    def __init__(self, config: Qwen2Config):
        super(LlavaQWen2Model, self).__init__(config)

class LlavaQWen2ForCausalLM(Qwen2ForCausalLM, LlavaMetaForCausalLM): ##FIXME: use LLavaMetaForCausalLM here, don't konw if need LLaVAQWenMetaForCausalLM
    config_class = LlavaQWen2Config

    def __init__(self, config):
        super(Qwen2ForCausalLM, self).__init__(config)
        self.model = LlavaQWen2Model(config)
        self.vocab_size = config.vocab_size
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

        self.post_init()

    def get_model(self):
        return self.model

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
        cache_position: Optional[torch.LongTensor] = None,
        logits_to_keep: Union[int, torch.Tensor] = 0,
    ) -> Union[Tuple, CausalLMOutputWithPast]:
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

        
        out = super().forward(
                input_ids=input_ids,
                past_key_values=past_key_values,
                attention_mask=attention_mask,
                token_type_ids=token_type_ids,
                position_ids=position_ids,
                head_mask=head_mask,
                inputs_embeds=inputs_embeds,
                encoder_hidden_states=encoder_hidden_states,
                encoder_attention_mask=encoder_attention_mask,
                labels=labels,
                use_cache=use_cache,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict,
                cache_position = cache_position,
                logits_to_keep = logits_to_keep,
        )
        return out

    def prepare_inputs_for_generation(self, input_ids, past_key_values=None, inputs_embeds=None, **kwargs):
        # import ipdb
        # ipdb.set_trace()
        images = kwargs.pop("images", None)
        _inputs = super().prepare_inputs_for_generation(
            input_ids, past_key_values=past_key_values, inputs_embeds=inputs_embeds, **kwargs
        )
        if images is not None:
            _inputs['images'] = images
        return _inputs


AutoConfig.register("llava_qwen2", LlavaQWen2Config)
AutoTokenizer.register(LlavaQWen2Config, Qwen2Tokenizer)
AutoModelForCausalLM.register(LlavaQWen2Config, LlavaQWen2ForCausalLM)
