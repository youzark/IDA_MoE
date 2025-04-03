import os
import torch
import json

from torch.utils.data import Sampler

from transformers import Trainer
from transformers.trainer import (
    is_sagemaker_mp_enabled,
    get_parameter_names,
    has_length,
    ALL_LAYERNORM_LAYERS,
    # ShardedDDPOption,
    logger,
)
from typing import List, Optional
import torch.distributed as dist

import numpy as np
from scipy import stats


def maybe_zero_3(param, ignore_status=False, name=None):
    from deepspeed import zero
    from deepspeed.runtime.zero.partition_parameters import ZeroParamStatus
    if hasattr(param, "ds_id"):
        if param.ds_status == ZeroParamStatus.NOT_AVAILABLE:
            if not ignore_status:
                print(name, 'no ignore status')
        with zero.GatheredParameters([param]):
            param = param.data.detach().cpu().clone()
    else:
        param = param.detach().cpu().clone()
    return param

def gather_routing_entries(local_entries):
    """
    Gathers a list of routing entries from all DeepSpeed DDP processes.
    Each process provides its own list of routing entries.
    Returns a combined list on rank 0 and None on other ranks.
    """
    # If distributed training is not initialized, simply return local entries.
    if not (dist.is_available() and dist.is_initialized()):
        return local_entries

    world_size = dist.get_world_size()
    # Create a list placeholder to gather entries from all ranks.
    gathered_entries = [None] * world_size

    # all_gather_object will collect the local_entries (a list) from all processes.
    dist.all_gather_object(gathered_entries, local_entries)

    # Only on rank 0 do we merge all entries.
    if dist.get_rank() == 0:
        merged_entries = []
        for entries in gathered_entries:
            merged_entries.extend(entries)
        return merged_entries
    else:
        return None


def get_mm_adapter_state_maybe_zero_3(named_params, keys_to_match):
    to_return = {k: t for k, t in named_params if any(key_match in k for key_match in keys_to_match)}
    to_return = {k: maybe_zero_3(v, ignore_status=True, name=k).cpu() for k, v in to_return.items()}
    return to_return


def split_to_even_chunks(indices, lengths, num_chunks):
    """
    Split a list of indices into `chunks` chunks of roughly equal lengths.
    """

    if len(indices) % num_chunks != 0:
        return [indices[i::num_chunks] for i in range(num_chunks)]

    num_indices_per_chunk = len(indices) // num_chunks

    chunks = [[] for _ in range(num_chunks)]
    chunks_lengths = [0 for _ in range(num_chunks)]
    for index in indices:
        shortest_chunk = chunks_lengths.index(min(chunks_lengths))
        chunks[shortest_chunk].append(index)
        chunks_lengths[shortest_chunk] += lengths[index]
        if len(chunks[shortest_chunk]) == num_indices_per_chunk:
            chunks_lengths[shortest_chunk] = float("inf")

    return chunks


def get_modality_length_grouped_indices(lengths, batch_size, world_size, generator=None):
    # We need to use torch for the random part as a distributed sampler will set the random seed for torch.
    assert all(l != 0 for l in lengths), "Should not have zero length."
    if all(l > 0 for l in lengths) or all(l < 0 for l in lengths):
        # all samples are in the same modality
        return get_length_grouped_indices(lengths, batch_size, world_size, generator=generator)
    mm_indices, mm_lengths = zip(*[(i, l) for i, l in enumerate(lengths) if l > 0])
    lang_indices, lang_lengths = zip(*[(i, -l) for i, l in enumerate(lengths) if l < 0])

    mm_shuffle = [mm_indices[i] for i in get_length_grouped_indices(mm_lengths, batch_size, world_size, generator=None)]
    lang_shuffle = [lang_indices[i] for i in get_length_grouped_indices(lang_lengths, batch_size, world_size, generator=None)]
    megabatch_size = world_size * batch_size
    mm_megabatches = [mm_shuffle[i : i + megabatch_size] for i in range(0, len(mm_shuffle), megabatch_size)]
    lang_megabatches = [lang_shuffle[i : i + megabatch_size] for i in range(0, len(lang_shuffle), megabatch_size)]

    last_mm = mm_megabatches[-1]
    last_lang = lang_megabatches[-1]
    additional_batch = last_mm + last_lang
    megabatches = mm_megabatches[:-1] + lang_megabatches[:-1]
    megabatch_indices = torch.randperm(len(megabatches), generator=generator)
    megabatches = [megabatches[i] for i in megabatch_indices]

    if len(additional_batch) > 0:
        megabatches.append(sorted(additional_batch))

    return [i for megabatch in megabatches for i in megabatch]


def get_length_grouped_indices(lengths, batch_size, world_size, generator=None, merge=True):
    # We need to use torch for the random part as a distributed sampler will set the random seed for torch.
    indices = torch.randperm(len(lengths), generator=generator)
    megabatch_size = world_size * batch_size
    megabatches = [indices[i : i + megabatch_size].tolist() for i in range(0, len(lengths), megabatch_size)]
    megabatches = [sorted(megabatch, key=lambda i: lengths[i], reverse=True) for megabatch in megabatches]
    megabatches = [split_to_even_chunks(megabatch, lengths, world_size) for megabatch in megabatches]

    return [i for megabatch in megabatches for batch in megabatch for i in batch]


class LengthGroupedSampler(Sampler):
    r"""
    Sampler that samples indices in a way that groups together features of the dataset of roughly the same length while
    keeping a bit of randomness.
    """

    def __init__(
        self,
        batch_size: int,
        world_size: int,
        lengths: Optional[List[int]] = None,
        generator=None,
        group_by_modality: bool = False,
    ):
        if lengths is None:
            raise ValueError("Lengths must be provided.")

        self.batch_size = batch_size
        self.world_size = world_size
        self.lengths = lengths
        self.generator = generator
        self.group_by_modality = group_by_modality

    def __len__(self):
        return len(self.lengths)

    def __iter__(self):
        if self.group_by_modality:
            indices = get_modality_length_grouped_indices(self.lengths, self.batch_size, self.world_size, generator=self.generator)
        else:
            indices = get_length_grouped_indices(self.lengths, self.batch_size, self.world_size, generator=self.generator)
        return iter(indices)


class LLaVATrainer(Trainer):
    def __init__(
            self,
            *args,
            routing_file_path: str|None = None,
            training:bool = True,
            **kwargs,
    ):
        # Add memory tracking before init
        if torch.npu.is_available():
            print(f"Before trainer init - Memory allocated: {torch.npu.memory_allocated() / 1e9:.2f} GB")
            print(f"Before trainer init - Memory reserved: {torch.npu.memory_reserved() / 1e9:.2f} GB")

        super().__init__(*args,**kwargs)

        # Add memory tracking after init
        if torch.npu.is_available():
            print(f"After trainer init - Memory allocated: {torch.npu.memory_allocated() / 1e9:.2f} GB")
            print(f"After trainer init - Memory reserved: {torch.npu.memory_reserved() / 1e9:.2f} GB")
        
        self._if_training = training
        self.routing_file_path = routing_file_path
        self._if_doc_routing = routing_file_path.lower().endswith('.jsonl')


    def _get_train_sampler(self) -> Optional[torch.utils.data.Sampler]:
        if self.train_dataset is None or not has_length(self.train_dataset):
            return None

        if self.args.group_by_modality_length:
            lengths = self.train_dataset.modality_lengths
            return LengthGroupedSampler(
                self.args.train_batch_size,
                world_size=self.args.world_size * self.args.gradient_accumulation_steps,
                lengths=lengths,
                group_by_modality=True,
            )
        else:
            return super()._get_train_sampler()

    def create_optimizer(self):
        """
        Setup the optimizer.

        We provide a reasonable default that works well. If you want to use something else, you can pass a tuple in the
        Trainer's init through `optimizers`, or subclass and override this method in a subclass.
        """
        if is_sagemaker_mp_enabled():
            return super().create_optimizer()
        # if self.sharded_ddp == ShardedDDPOption.SIMPLE:
        #     return super().create_optimizer()

        opt_model = self.model

        if self.optimizer is None:
            decay_parameters = get_parameter_names(opt_model, ALL_LAYERNORM_LAYERS)
            decay_parameters = [name for name in decay_parameters if "bias" not in name]
            if self.args.mm_projector_lr is not None:
                projector_parameters = [name for name, _ in opt_model.named_parameters() if "mm_projector" in name]
                optimizer_grouped_parameters = [
                    {
                        "params": [
                            p for n, p in opt_model.named_parameters() if (n in decay_parameters and n not in projector_parameters and p.requires_grad)
                        ],
                        "weight_decay": self.args.weight_decay,
                        "name": "decay_no_proj_parameters"
                    },
                    {
                        "params": [
                            p for n, p in opt_model.named_parameters() if (n not in decay_parameters and n not in projector_parameters and p.requires_grad)
                        ],
                        "weight_decay": 0.0,
                        "name": "no_decay_no_proj_parameters"
                    },
                    {
                        "params": [
                            p for n, p in opt_model.named_parameters() if (n in decay_parameters and n in projector_parameters and p.requires_grad)
                        ],
                        "weight_decay": self.args.weight_decay,
                        "lr": self.args.mm_projector_lr,
                        "name": "decay_proj_parameters"
                    },
                    {
                        "params": [
                            p for n, p in opt_model.named_parameters() if (n not in decay_parameters and n in projector_parameters and p.requires_grad)
                        ],
                        "weight_decay": 0.0,
                        "lr": self.args.mm_projector_lr,
                        "name": "no_decay_proj_parameters"
                    },
                ]
            else:
                gmm_parameters = [name for name, _ in opt_model.named_parameters() if "means" in name or "vars" in name or "mix_logits" in name]
                optimizer_grouped_parameters = [
                    {
                        "params": [
                            p for n, p in opt_model.named_parameters() if (n in decay_parameters and p.requires_grad and n not in gmm_parameters) 
                        ],
                        "weight_decay": self.args.weight_decay,
                        "name": "decay_parameters"
                    },
                    {
                        "params": [
                            p for n, p in opt_model.named_parameters() if (n not in decay_parameters and p.requires_grad and n not in gmm_parameters)
                        ],
                        "weight_decay": 0.0,
                        "name": "no_decay_parameters"
                    },
                    {
                        "params": [
                            p for n, p in opt_model.named_parameters() 
                            if (n in gmm_parameters and p.requires_grad)
                        ],
                        "weight_decay": 0.0,
                        "lr": 0.01,
                        "name": "gmm_parameters"
                    },
                ]

            optimizer_grouped_parameters = [
                group for group in optimizer_grouped_parameters 
                if len(group["params"]) > 0
            ]

            if self.args.moe_enable:
                from deepspeed.moe.utils import split_params_into_different_moe_groups_for_optimizer
                optimizer_grouped_parameters = split_params_into_different_moe_groups_for_optimizer(optimizer_grouped_parameters)
            optimizer_cls, optimizer_kwargs = Trainer.get_optimizer_cls_and_kwargs(self.args)

            self.optimizer = optimizer_cls(optimizer_grouped_parameters, **optimizer_kwargs)

            # for i, group in enumerate(self.optimizer.param_groups):
            #     print(f"\nParameter group {i}:")
            #     print(f"Learning rate: {group['lr']}")
            #     print(f"Weight decay: {group.get('weight_decay', 0)}")
            #     print(f"Other params: {[(k,v) for k,v in group.items() if k not in ['params', 'lr', 'weight_decay']]}")
            #     print(f"Number of parameters: {len(group['params'])}")


            if optimizer_cls.__name__ == "Adam8bit":
                import bitsandbytes
                manager = bitsandbytes.optim.GlobalOptimManager.get_instance()

                skipped = 0
                for module in opt_model.modules():
                    if isinstance(module, nn.Embedding):
                        skipped += sum({p.data_ptr(): p.numel() for p in module.parameters()}.values())
                        logger.info(f"skipped {module}: {skipped/2**20}M params")
                        manager.register_module_override(module, "weight", {"optim_bits": 32})
                        logger.debug(f"bitsandbytes: will optimize {module} in fp32")
                logger.info(f"skipped: {skipped/2**20}M params")

        return self.optimizer

    def _save_checkpoint(self, model, trial, metrics=None):
        if getattr(self.args, 'tune_mm_mlp_adapter', False):
            from transformers.trainer_utils import PREFIX_CHECKPOINT_DIR
            checkpoint_folder = f"{PREFIX_CHECKPOINT_DIR}-{self.state.global_step}"

            run_dir = self._get_output_dir(trial=trial)
            output_dir = os.path.join(run_dir, checkpoint_folder)

            # Only save Adapter
            keys_to_match = ['mm_projector', 'vision_resampler']
            if getattr(self.args, "use_im_start_end", False):
                keys_to_match.extend(['embed_tokens', 'embed_in'])

            weight_to_save = get_mm_adapter_state_maybe_zero_3(self.model.named_parameters(), keys_to_match)

            if self.args.local_rank == 0 or self.args.local_rank == -1:
                self.model.config.save_pretrained(output_dir)
                torch.save(weight_to_save, os.path.join(output_dir, f'mm_projector.bin'))
        else:
            super(LLaVATrainer, self)._save_checkpoint(model, trial, metrics)

    def _save(self, output_dir: Optional[str] = None, state_dict=None):
        if getattr(self.args, 'tune_mm_mlp_adapter', False):
            pass
        else:
            super(LLaVATrainer, self)._save(output_dir, state_dict)


    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        # Get loss and outputs from parent implementation
        ids = inputs["ids"]
        inputs.pop("ids")
        loss, outputs = super().compute_loss(model, inputs, return_outputs=True, num_items_in_batch = num_items_in_batch)

        zero_loss = torch.tensor(0.0, device=outputs["loss"].device, requires_grad=True)


        # Add your custom logging
        if self.args.moe_enable and self.args.local_rank <= 0:  # only log on main process
            cvs = [ metrics["cv"] for metrics in outputs["moe_metrics_list"] if metrics and "cv" in metrics]
            entropy = [ metrics["gating_Entropy"] for metrics in outputs["moe_metrics_list"] if metrics and "gating_Entropy" in metrics]
            avg_ent = torch.stack(entropy).mean()
            try:
                pearson_r = stats.pearsonr(avg_ent.float().cpu().detach().numpy(), outputs["avg_ppl"].float().cpu().detach().numpy())[0]
                if np.isnan(pearson_r):
                    pearson_r = 0.0
            except:
                pearson_r = 0.0
            self.accum_cor = self.accum_cor + pearson_r if hasattr(self,"accum_cor") else pearson_r
            self.accum_count = self.accum_count + 1 if hasattr(self,"accum_count") else 1


            self.log({
                "train/moe_loss": outputs["moe_loss"].item(),
                "train/perplexity": torch.exp(outputs["ce_loss"]).item(),  # Just the CE loss
                "train/worst_cv": max(cvs),
                "train/mean_cv": sum(cvs) / len(cvs),
                "train/entropy_ppl_correlation": self.accum_cor / self.accum_count,
                "train/mean_ent": avg_ent.item(),
            })
        
        if self._if_doc_routing:
            """
            log the routing decision of the model so that we can compare how stable the training is gaven data
            """
            local_routing_logs = []
            for tt, id_ in enumerate(ids):
                routing_entry = {
                    "routing_decision": [
                        # For example, if you want the index of the max logit.
                        layer_metric["gating_logits"][tt, :, :].max(dim=-1)[1].tolist()
                        for layer_metric in outputs["moe_metrics_list"]
                        if layer_metric and "gating_logits" in layer_metric
                    ],
                    "sample_ids": id_,
                }
                local_routing_logs.append(routing_entry)
            # Gather routing logs from all DeepSpeed processes.
            merged_entries = gather_routing_entries(local_routing_logs)
            # Write the merged entries to a single JSONL file only on rank 0.
            if self.args.local_rank == 0 and merged_entries is not None:
                with open(self.routing_file_path, "a") as f:
                    for entry in merged_entries:
                        f.write(json.dumps(entry) + "\n")
        
        if self._if_training:
            return (loss, outputs) if return_outputs else loss
        else:
            return (zero_loss, outputs) if return_outputs else zero_loss

    # def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
    #     # Initialize batch counter if not exists
    #     if not hasattr(self, "_memory_analysis_batch_counter"):
    #         self._memory_analysis_batch_counter = 0
        
    #     # Run memory analysis every 30 batches
    #     run_analysis = torch.npu.is_available() and (
    #         self._memory_analysis_batch_counter % 3 == 0 or 
    #         not hasattr(self, "_memory_analysis_run_count")
    #     )
        
    #     if run_analysis:
    #         # Initialize run count if not exists
    #         if not hasattr(self, "_memory_analysis_run_count"):
    #             self._memory_analysis_run_count = 0
            
    #         # Increment run count
    #         self._memory_analysis_run_count += 1
    #         run_number = self._memory_analysis_run_count
            
    #         print(f"\n=== MEMORY ANALYSIS (BATCH {self._memory_analysis_batch_counter}, RUN {run_number}) ===")
            
    #         # Track basic metrics
    #         total_memory = torch.npu.memory_allocated() / 1e9
    #         reserved_memory = torch.npu.memory_reserved() / 1e9
    #         peak_memory = torch.npu.max_memory_allocated() / 1e9
            
    #         print(f"PyTorch memory: {total_memory:.2f} GB allocated, {reserved_memory:.2f} GB reserved, {peak_memory:.2f} GB peak")
            
    #         # Try to get system-level memory info
    #         try:
    #             import os
    #             import psutil
    #             process = psutil.Process(os.getpid())
    #             system_memory = process.memory_info().rss / 1e9
    #             print(f"System memory: {system_memory:.2f} GB used by process")
    #             print(f"Memory gap: {system_memory - total_memory:.2f} GB untracked by PyTorch")
    #         except Exception as e:
    #             print(f"Could not get system memory info: {e}")
            
    #         # DeepSpeed internal memory analysis
    #         if self.is_deepspeed_enabled and hasattr(self, "deepspeed"):
    #             ds = self.deepspeed
                
    #             print("\nDEEPSPEED BUFFER ANALYSIS:")
    #             # Check for large buffers
    #             try:
    #                 # Common places where DeepSpeed allocates large buffers
    #                 buffer_locations = [
    #                     'zero_grad_buffer', 'buckets', 'reduce_buckets', 'param_buffers',
    #                     'gradient_accumulation_buffer', 'reduce_scatter_buffer', 'all_gather_buffer',
    #                     'temp_contiguous_buffer'
    #                 ]
                    
    #                 total_buffer_size = 0
    #                 for buffer_name in buffer_locations:
    #                     if hasattr(ds, buffer_name):
    #                         buffer = getattr(ds, buffer_name)
    #                         buffer_size = 0
                            
    #                         if torch.is_tensor(buffer):
    #                             buffer_size = buffer.numel() * buffer.element_size()
    #                         elif isinstance(buffer, list):
    #                             for item in buffer:
    #                                 if torch.is_tensor(item):
    #                                     buffer_size += item.numel() * item.element_size()
                            
    #                         buffer_size_gb = buffer_size / 1e9
    #                         if buffer_size_gb > 0.1:  # Only show significant buffers
    #                             print(f"{buffer_name}: {buffer_size_gb:.2f} GB")
    #                         total_buffer_size += buffer_size
                    
    #                 print(f"Total identified buffer size: {total_buffer_size / 1e9:.2f} GB")
    #             except Exception as e:
    #                 print(f"Error analyzing DeepSpeed buffers: {e}")
                
    #             # Check optimizer memory
    #             try:
    #                 optimizer_memory = 0
                    
    #                 if hasattr(ds, "optimizer") and ds.optimizer is not None:
    #                     print("\nOPTIMIZER ANALYSIS:")
                        
    #                     # FP32 master weights (main memory consumer in ZeRO)
    #                     if hasattr(ds.optimizer, "fp32_partitioned_groups_flat"):
    #                         fp32_size = 0
    #                         for group in ds.optimizer.fp32_partitioned_groups_flat:
    #                             if torch.is_tensor(group):
    #                                 fp32_size += group.numel() * group.element_size()
    #                         fp32_size_gb = fp32_size / 1e9
    #                         print(f"FP32 partitioned groups: {fp32_size_gb:.2f} GB")
    #                         optimizer_memory += fp32_size
                        
    #                     # FP16 partitioned groups
    #                     if hasattr(ds.optimizer, "fp16_partitioned_groups_flat"):
    #                         fp16_size = 0
    #                         for group in ds.optimizer.fp16_partitioned_groups_flat:
    #                             if torch.is_tensor(group):
    #                                 fp16_size += group.numel() * group.element_size()
    #                         fp16_size_gb = fp16_size / 1e9
    #                         print(f"FP16 partitioned groups: {fp16_size_gb:.2f} GB")
    #                         optimizer_memory += fp16_size
                        
    #                     # Optimizer states
    #                     state_size = 0
    #                     state_count = 0
    #                     if hasattr(ds.optimizer, "optimizer") and hasattr(ds.optimizer.optimizer, "state"):
    #                         for param_id, param_state in ds.optimizer.optimizer.state.items():
    #                             for state_name, state_value in param_state.items():
    #                                 if torch.is_tensor(state_value):
    #                                     state_size += state_value.numel() * state_value.element_size()
    #                                     state_count += 1
                            
    #                         state_size_gb = state_size / 1e9
    #                         print(f"Optimizer states: {state_count} tensors, {state_size_gb:.2f} GB")
    #                         optimizer_memory += state_size
                        
    #                     print(f"Total optimizer memory found: {optimizer_memory / 1e9:.2f} GB")
    #             except Exception as e:
    #                 print(f"Error analyzing optimizer: {e}")
            
    #         # MoE Analysis
    #         if self.args.moe_enable:
    #             print("\nMoE MEMORY ANALYSIS:")
    #             try:
    #                 # Check activation memory in MoE modules
    #                 moe_activation_size = 0
                    
    #                 for name, module in model.named_modules():
    #                     if "moe" in name.lower() or "expert" in name.lower():
    #                         for attr_name in dir(module):
    #                             if attr_name.startswith('_'): continue
    #                             try:
    #                                 attr = getattr(module, attr_name)
    #                                 if torch.is_tensor(attr) and not isinstance(attr, torch.nn.Parameter):
    #                                     moe_activation_size += attr.numel() * attr.element_size()
    #                             except:
    #                                 pass
                    
    #                 print(f"MoE activation tensors: {moe_activation_size / 1e9:.2f} GB")
                    
    #                 # Look for any large routing tensors in the outputs
    #                 if return_outputs and 'moe_metrics_list' in outputs:
    #                     routing_size = 0
    #                     for layer_metrics in outputs['moe_metrics_list']:
    #                         if layer_metrics and 'gating_logits' in layer_metrics:
    #                             tensor = layer_metrics['gating_logits']
    #                             routing_size += tensor.numel() * tensor.element_size()
                        
    #                     print(f"MoE routing tensors in current batch: {routing_size / 1e9:.2f} GB")
    #             except Exception as e:
    #                 print(f"Error analyzing MoE memory: {e}")
            
    #         # Memory growth analysis (after first run)
    #         if hasattr(self, "_previous_memory_analysis"):
    #             prev = self._previous_memory_analysis
    #             memory_growth = total_memory - prev["total_memory"]
    #             system_growth = 0
    #             if "system_memory" in prev and "system_memory" in locals():
    #                 system_growth = system_memory - prev["system_memory"]
                
    #             print("\nMEMORY GROWTH SINCE LAST ANALYSIS:")
    #             print(f"PyTorch tracked memory growth: {memory_growth:.2f} GB")
    #             if "system_memory" in prev and "system_memory" in locals():
    #                 print(f"System memory growth: {system_growth:.2f} GB")
                
    #             # Check for potential memory leak
    #             if run_number > 3 and system_growth > 2.0:
    #                 print("\n⚠️ POTENTIAL MEMORY LEAK DETECTED ⚠️")
    #                 print(f"Memory increased by {system_growth:.2f} GB over {30} batches")
    #                 print("Consider implementing explicit memory cleanup or checking for tensor retention")
            
    #         # Store current values for next comparison
    #         self._previous_memory_analysis = {
    #             "total_memory": total_memory,
    #             "reserved_memory": reserved_memory,
    #             "peak_memory": peak_memory
    #         }
    #         if "system_memory" in locals():
    #             self._previous_memory_analysis["system_memory"] = system_memory
            
    #         # Reset peak memory stats for next window
    #         torch.npu.reset_peak_memory_stats()
        
    #     # Increment batch counter
    #     self._memory_analysis_batch_counter += 1
        
    #     # Original compute_loss implementation
    #     ids = inputs.pop("ids", None)
    #     loss, outputs = super().compute_loss(model, inputs, return_outputs=True, num_items_in_batch=num_items_in_batch)
        
    #     # Rest of your compute_loss method implementation
    #     zero_loss = torch.tensor(0.0, device=outputs["loss"].device, requires_grad=True)
        
    #     # Your MoE logging code
    #     if self.args.moe_enable and self.args.local_rank <= 0:
    #         # [Your existing MoE logging code]
    #         pass
        
    #     if self._if_doc_routing and ids is not None:
    #         # [Your existing routing code]
    #         pass
        
    #     if self._if_training:
    #         return (loss, outputs) if return_outputs else loss
    #     else:
    #         return (zero_loss, outputs) if return_outputs else zero_loss