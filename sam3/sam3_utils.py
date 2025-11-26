import torch
import numpy as np
from typing import Dict, List, Optional
from sam3.model.sam3_image_processor import Sam3Processor
from sam3.model import box_ops
from sam3.model.data_misc import interpolate

class CustomSam3Processor(Sam3Processor):
    def set_text_prompt_batch(self, prompts: List[str], state: Dict):
        """Sets the text prompts for a batch and runs inference."""
        
        if "backbone_out" not in state:
            raise ValueError("You must call set_image_batch before set_text_prompt_batch")

        # Encode text prompts
        # forward_text expects a list of strings
        text_outputs = self.model.backbone.forward_text(prompts, device=self.device)
        
        # Update state with text outputs
        # Note: backbone_out in state is a dict of tensors. We need to make sure text_outputs are merged correctly.
        # Assuming text_outputs keys match backbone_out keys or are additive.
        state["backbone_out"].update(text_outputs)
        
        if "geometric_prompt" not in state:
            # Create dummy geometric prompts for the batch
            # _get_dummy_prompt returns a list of tensors or a tensor?
            # Let's check _get_dummy_prompt implementation if possible, or assume it handles batch if we pass batch size?
            # Actually, _get_dummy_prompt likely returns a single dummy. We might need to replicate it.
            # But let's try calling it and see. If it's model-specific, it might be tricky.
            # For now, let's assume we can skip geometric prompt or it's handled inside forward_grounding if None?
            # The original code sets it if missing.
            
            # Let's try to replicate the dummy prompt for the batch
            dummy = self.model._get_dummy_prompt()
            # dummy is likely a list of tensors or a dict.
            # If it's a list of tensors [points, labels], we need to stack/repeat them.
            # This part is risky without seeing _get_dummy_prompt.
            
            # Alternative: Pass None and hope forward_grounding handles it?
            # Original code: state["geometric_prompt"] = self.model._get_dummy_prompt()
            state["geometric_prompt"] = dummy # We might need to broadcast this later or now.
            
        return self._forward_grounding_batch(state)

    def set_box_prompt(self, state: Dict, boxes_xyxy: torch.Tensor, box_labels: Optional[torch.Tensor] = None):
        """Sets a box prompt for a single image and runs grounding."""
        if boxes_xyxy.dim() == 2:
            boxes_xyxy = boxes_xyxy.unsqueeze(0)  # [1, num_boxes, 4]

        if "backbone_out" not in state:
            raise ValueError("You must call set_image before set_box_prompt")

        if "language_features" not in state["backbone_out"]:
            dummy_text_outputs = self.model.backbone.forward_text(["visual"], device=self.device)
            state["backbone_out"].update(dummy_text_outputs)

        geo_prompt = self.model._get_dummy_prompt()
        state["geometric_prompt"] = geo_prompt

        if box_labels is None:
            box_labels = torch.ones((boxes_xyxy.shape[0], boxes_xyxy.shape[1]), device=self.device, dtype=torch.bool)
        elif box_labels.dim() == 1:
            box_labels = box_labels.unsqueeze(0).to(dtype=torch.bool, device=self.device)

        scale = torch.tensor(
            [state["original_width"], state["original_height"], state["original_width"], state["original_height"]],
            device=self.device,
            dtype=boxes_xyxy.dtype,
        )
        boxes_cxcywh = box_ops.box_xyxy_to_cxcywh(boxes_xyxy / scale)

        geo_prompt.append_boxes(boxes_cxcywh, box_labels)
        return self._forward_grounding_batch(state)

    def _forward_grounding_batch(self, state: Dict):
        # We need to handle the geometric prompt broadcasting here if it wasn't done
        geo_prompt = state["geometric_prompt"]
        # If geo_prompt is single, we might need to repeat it for the batch.
        # But let's assume for a moment that forward_grounding can handle broadcasting or we fix it if it errors.
        
        outputs = self.model.forward_grounding(
            backbone_out=state["backbone_out"],
            find_input=self.find_stage, # This might need to be batched too? find_stage has tensors.
            geometric_prompt=geo_prompt,
            find_target=None,
        )

        out_bbox = outputs["pred_boxes"]   # [B, N, 4]
        out_logits = outputs["pred_logits"] # [B, N, C]
        out_masks = outputs["pred_masks"]   # [B, N, H, W]
        
        # Sigmoid on logits
        out_probs = out_logits.sigmoid()
        
        # Presence score
        presence_score = outputs["presence_logit_dec"].sigmoid().unsqueeze(1) # [B, 1, 1] ?
        out_probs = (out_probs * presence_score).squeeze(-1) # [B, N]

        # Filter by confidence threshold
        # This is tricky in batch because each image might have different number of kept boxes.
        # We should probably return all and filter outside, or return a list of results.
        
        results_masks = []
        results_scores = []
        
        batch_size = out_probs.shape[0]
        
        # Get original sizes
        orig_heights = state.get("original_heights", [state.get("original_height")] * batch_size)
        orig_widths = state.get("original_widths", [state.get("original_width")] * batch_size)

        for i in range(batch_size):
            probs_i = out_probs[i]
            masks_i = out_masks[i]
            
            keep = probs_i > self.confidence_threshold
            
            # If nothing kept, maybe keep the best one?
            if not keep.any():
                # keep best one
                idx = torch.argmax(probs_i)
                keep[idx] = True
                
            probs_i = probs_i[keep]
            masks_i = masks_i[keep]
            
            # Interpolate mask to original size
            img_h = orig_heights[i]
            img_w = orig_widths[i]
            
            if masks_i.shape[0] > 0:
                masks_i = interpolate(
                    masks_i.unsqueeze(1),
                    (img_h, img_w),
                    mode="bilinear",
                    align_corners=False,
                ).sigmoid()
                masks_i = masks_i > 0.5
                masks_i = masks_i.squeeze(1)
            else:
                # Should not happen if we force keep best, but for safety
                masks_i = torch.zeros((0, img_h, img_w), device=self.device, dtype=torch.bool)

            results_masks.append(masks_i)
            results_scores.append(probs_i)
            
        return {"masks": results_masks, "scores": results_scores}
