def generate_with_moe_metrics(model, input_ids, images=None, **generate_kwargs):
    """
    Run model.generate() and return both the generation output and MoE metrics from the last turn.
    """
    # Container to store the last MoE metrics
    last_moe_metrics = [None]
    
    # Define hook to capture MoE metrics
    def capture_last_moe_metrics(module, inputs, outputs):
        if hasattr(outputs, "moe_metrics_list") and outputs.moe_metrics_list is not None:
            last_moe_metrics[0] = outputs.moe_metrics_list
        return outputs
    
    hook = model.register_forward_hook(capture_last_moe_metrics)
    
    try:
        # Run generation with all the provided kwargs
        output_dict = model.generate(
            input_ids,
            images=images,
            return_dict_in_generate=True,  # Ensure this is True
            **generate_kwargs
        )
        
        # Add the captured metrics to the output dictionary
        # Convert to regular dict if it's not already
        if not isinstance(output_dict, dict):
            output_dict = output_dict._asdict()
        
        # Add the MoE metrics to the output
        output_dict['moe_metrics_list'] = last_moe_metrics[0]
        
        return output_dict
        
    finally:
        # Always remove the hook to prevent memory leaks
        hook.remove()