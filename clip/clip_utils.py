import clip



def load_clip_model(backbone: str, device: str):
    
    model, processor = clip.load(backbone, device=device)
    

    return model, processor




def setup_trainable_parameters(clip_model, logger, train_mode, grad_from_block):
    for m in clip_model.parameters():
        m.requires_grad = False

    if train_mode == 'last_vision_block':
        assert grad_from_block is not None, "grad_from_block must be specified for last_vision_block training mode"
        assert grad_from_block == len(clip_model.visual.transformer.resblocks) - 1, "grad_from_block must be set to the last block index for last_vision_block training mode"

        logger.info("Unfreezing last vision block")
        for name, m in clip_model.named_parameters():
            if 'visual.transformer.resblocks' in name:
                block_num = int(name.split('.')[3])
                if block_num >= grad_from_block:
                    m.requires_grad = True      
    else:
        raise ValueError(f"Unsupported CLIP training mode implementation: {train_mode}")

    return clip_model

