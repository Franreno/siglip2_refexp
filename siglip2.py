import torch
import torch.nn as nn
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
from typing import Tuple, Union, List
import torch.nn.functional as F
from torchvision.ops import generalized_box_iou, box_iou

from transformers import (
    Siglip2Model,
    Siglip2TextModel,
    Siglip2VisionModel,
    Siglip2Processor,
)
from transformers.modeling_outputs import BaseModelOutputWithPooling


class Siglip2Encoders(nn.Module):
    """
    A wrapper around SigLIP 2's text and vision encoders using Hugging Face Transformers.

    This class provides utilities to encode images and text independently or together, using pretrained
    SigLIP 2 submodules. It supports returning either pooled representations or full token/patch embeddings.

    Attributes:
        processor (Siglip2Processor): Combines text tokenizer and image processor.
        text_encoder (Siglip2TextModel): Pretrained transformer-based text encoder.
        vision_encoder (Siglip2VisionModel): Pretrained vision transformer (ViT) image encoder.
        device (str): Computation device (defaults to 'cuda' if available, otherwise 'cpu').

    Example:
        >>> model = Siglip2Encoders()
        >>> img = Image.open("cat.jpg").convert("RGB")
        >>> text_emb = model.encode_text("a cat")
        >>> img_emb = model.encode_image(img)
    """

    def __init__(self,
                 model_name: str = "google/siglip2-base-patch16-naflex",
                 device: str = None):
        """
        Initialize Siglip2Encoders with pretrained vision and text models.
        Some models_name result on this error:
            >You are using a model of type siglip_text_model to instantiate a 
            >model of type siglip2_text_model. This is not supported for all 
            >configurations of models and can yield errors.

        Therefore, this init loads the full model first then extracts the text
        and vision encoders only.

        Args:
            model_name (str): The model checkpoint to load (e.g., 'google/siglip2-base-patch16-naflex').
            device (str, optional): Device to run the model on ('cuda' or 'cpu'). If None, auto-detects.
        """
        super().__init__()

        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")

        # Unified processor for both text and image (includes tokenizer + image preprocessor)
        self.processor = Siglip2Processor.from_pretrained(model_name)

        # Load full model
        full_model = Siglip2Model.from_pretrained(model_name)

        # Extract submodules
        self.text_encoder = full_model.text_model
        self.vision_encoder = full_model.vision_model

        # Move everything to the correct device
        self.to(self.device)

    def encode_text(self,
                    texts: Union[str, List[str]],
                    return_pooled: bool = True) -> torch.Tensor:
        """
        Encode text input(s) into embeddings using the SigLIP 2 text encoder.

        Args:
            texts (str or List[str]): A single string or a list of text inputs.
            return_pooled (bool): If True, return pooled [CLS] representation. Otherwise, return full token-level output.

        Returns:
            torch.Tensor: Embeddings of shape (batch_size, hidden_dim) if pooled, or (batch_size, seq_len, hidden_dim).
        """
        if isinstance(texts, str):
            texts = [texts]

        # Tokenize and format input
        inputs = self.processor(text=texts,
                                return_tensors="pt",
                                padding=True,
                                truncation=True).to(self.device)

        # Encode text
        outputs: Union[Tuple,
                       BaseModelOutputWithPooling] = self.text_encoder(**inputs)

        return outputs.pooler_output if return_pooled else outputs.last_hidden_state

    def encode_image(self,
                     images: Union[Image.Image, List[Image.Image]],
                     return_pooled: bool = True) -> torch.Tensor:
        """
        Encode image(s) into embeddings using the SigLIP 2 vision encoder.

        Args:
            images (PIL.Image or List[PIL.Image]): A single image or a list of PIL Image objects.
            return_pooled (bool): If True, return pooled [CLS] representation. Otherwise, return full patch-level output.

        Returns:
            torch.Tensor: Embeddings of shape (batch_size, hidden_dim) if pooled, or (batch_size, seq_len, hidden_dim).
        """
        if isinstance(images, Image.Image):
            images = [images]

        # Preprocess image(s)
        inputs = self.processor(images=images,
                                return_tensors="pt").to(self.device)

        # Rename fields to match Siglip2VisionTransformer signature
        vision_inputs = {
            "pixel_values": inputs["pixel_values"],
            "attention_mask": inputs.get("pixel_attention_mask"),
            "spatial_shapes": inputs.get("spatial_shapes"),
        }

        # Encode image
        outputs: Union[Tuple, BaseModelOutputWithPooling] = self.vision_encoder(
            **vision_inputs)

        return outputs.pooler_output if return_pooled else outputs.last_hidden_state

    def forward(
            self,
            texts: Union[str, List[str]],
            images: Union[Image.Image, List[Image.Image]],
            return_pooled: bool = True) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass to jointly encode both text and image inputs.

        Args:
            texts (str or List[str]): A single string or list of text inputs.
            images (PIL.Image or List[PIL.Image]): A single image or list of images.
            return_pooled (bool): If True, return pooled representations from both encoders.

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: (text_embeddings, image_embeddings)
        """
        text_embeds = self.encode_text(texts, return_pooled=return_pooled)
        image_embeds = self.encode_image(images, return_pooled=return_pooled)
        return text_embeds, image_embeds

    @torch.no_grad()
    def visualize_attention_map(self, image: Image.Image, save_path=None):
        """
        Visualizes the self-attention from the CLS token in the last transformer block
        as a heatmap overlaid on the original image.

        Args:
            image (PIL.Image): The image to visualize.
            show (bool): Whether to display the plot.
            save_path (str or None): If set, saves the image to this path.

        Returns:
            torch.Tensor: The attention heatmap of shape (H, W).
        """
        self.eval()

        # Preprocess
        inputs = self.processor(images=image,
                                return_tensors="pt").to(self.device)

        pixel_values = inputs["pixel_values"]
        attention_mask = inputs.get("pixel_attention_mask")
        spatial_shapes = inputs.get("spatial_shapes")
        if spatial_shapes is None:
            raise ValueError("Missing spatial_shapes in processor output.")

        h, w = spatial_shapes[0].tolist()

        vit_model = self.vision_encoder
        # Forward pass
        outputs = vit_model(
            pixel_values=pixel_values,
            attention_mask=attention_mask,
            spatial_shapes=spatial_shapes,
            return_dict=True,
            output_attentions=True,
        )

        # Get attention from the last layer
        attn = outputs.attentions[-1]  # [batch, heads, tokens, tokens]
        attn = attn.mean(dim=1)  # avg over heads
        num_patches = h * w
        cls_attn = attn[0, 0, -num_patches:]  # only keep last N values

        # Reshape attention to spatial
        h, w = spatial_shapes[0].tolist()
        heatmap = cls_attn.reshape(h, w).detach().cpu()
        heatmap = heatmap / heatmap.max()

        # Resize heatmap to image size
        heatmap_np = heatmap.numpy()
        heatmap_np = np.interp(heatmap_np, (heatmap_np.min(), heatmap_np.max()),
                               (0, 1))
        heatmap_np = np.uint8(255 * heatmap_np)
        heatmap_np = np.stack([heatmap_np] * 3, axis=-1)  # RGB

        # Upsample to image size
        heatmap_img = Image.fromarray(heatmap_np).resize(
            image.size, resample=Image.BILINEAR)
        heatmap_img = np.array(heatmap_img) / 255.0

        # Overlay
        fig, ax = plt.subplots()
        ax.imshow(image)
        ax.imshow(heatmap_img, cmap="jet", alpha=0.9)
        ax.axis("off")
        ax.set_title("Self-Attention Map (CLS → Patches)")

        if save_path:
            plt.savefig(save_path, bbox_inches="tight", pad_inches=0)

        return heatmap


class FusionTransformer(nn.Module):

    def __init__(
            self,
            hidden_dim=1536,  # Output size of google/siglip2-base-patch16-naflex (768)
            num_layers=2,
            num_heads=16,
            dropout=0.1):
        super().__init__()
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=num_heads,
            dim_feedforward=hidden_dim * 4,
            dropout=dropout,
            batch_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer,
                                                 num_layers=num_layers)

    def forward(self,
                combined_seq: torch.Tensor,
                attention_mask: torch.Tensor = None):
        """
        Args:
            combined_seq (torch.Tensor): Shape (batch_size, seq_len, hidden_dim)
            attention_mask (torch.Tensor, optional): Shape (batch_size, seq_len)
                Should contain 1s for valid tokens and 0s for padding.

        Returns:
            torch.Tensor: Encoded output of same shape (batch_size, seq_len, hidden_dim)
        """
        if attention_mask is not None:
            key_padding_mask = ~attention_mask.bool(
            )  # Transformer expects False for keep
        else:
            key_padding_mask = None

        return self.transformer(combined_seq,
                                src_key_padding_mask=key_padding_mask)


class BBoxPredictionHead(nn.Module):

    def __init__(self, hidden_dim=768):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 4),  # (x, y, w, h)
        )

    def forward(self, embeddings):
        """
        Args:
            text_cls (torch.Tensor): (batch_size, hidden_dim)
            image_cls (torch.Tensor): (batch_size, hidden_dim)

        Returns:
            torch.Tensor: Bounding box prediction (batch_size, 4)
        """
        bbox = self.fc(embeddings)
        return torch.sigmoid(bbox)


class RefExpLoss(nn.Module):
    """
    Combines Unbiased L1 loss and Generalized IoU loss for bounding box regression.

    Args:
        lambda_l1 (float): Weight for L1 loss component.
        lambda_iou (float): Weight for iou loss component.
        box_format (str): 'cxcywh' or 'xywh'. All boxes assumed normalized in [0, 1].
    """

    def __init__(
        self,
        lambda_l1=5.0,
        lambda_iou=1.0,
        box_format="xywh",
    ):
        super().__init__()
        assert box_format in ["cxcywh", "xywh"], "Unsupported box format"
        self.lambda_l1 = lambda_l1
        self.lambda_iou = lambda_iou
        self.box_format = box_format

    def forward(
        self,
        pred_boxes,
        target_boxes,
    ):
        """
        Args:
            pred_boxes (Tensor): shape (B, 4) - predicted boxes
            target_boxes (Tensor): shape (B, 4) - ground-truth boxes

        Returns:
            loss (Tensor): Scalar loss
            loss_dict (dict): {'l1': ..., 'iou': ...}
        """
        # Convert to xyxy for IoU
        pred_boxes_xyxy = self._box_to_xyxy(pred_boxes)
        target_boxes_xyxy = self._box_to_xyxy(target_boxes)

        # L1 Loss (unbiased)
        l1_loss = F.l1_loss(pred_boxes, target_boxes, reduction="mean")

        ious = box_iou(pred_boxes_xyxy, target_boxes_xyxy).diag()  # (B,)
        iou_loss = 1.0 - ious.mean()

        total_loss = self.lambda_l1 * l1_loss + self.lambda_iou * iou_loss
        loss_dict = {
            "l1": l1_loss.item(),
            "iou": iou_loss.item(),
            "total": total_loss.item(),
            "per_sample_iou": ious.detach().cpu(),
        }
        return total_loss, loss_dict

    def _box_to_xyxy(self, boxes):
        """
        Convert from [cx, cy, w, h] or [x, y, w, h] to [x1, y1, x2, y2]
        All inputs and outputs are normalized [0, 1].
        """
        if self.box_format == "cxcywh":
            cx, cy, w, h = boxes.unbind(-1)
            x1 = cx - 0.5 * w
            y1 = cy - 0.5 * h
            x2 = cx + 0.5 * w
            y2 = cy + 0.5 * h
        else:  # xywh
            x1, y1, w, h = boxes.unbind(-1)
            x2 = x1 + w
            y2 = y1 + h

        return torch.stack([x1, y1, x2, y2], dim=-1)


class RefExpModel(nn.Module):

    def __init__(
        self,
        siglip_model: Siglip2Encoders,
        fusion_transformer: FusionTransformer,
        bbox_head: BBoxPredictionHead,
        criterion: RefExpLoss,
    ):
        """
        Full referring expression model using frozen SIGLIP, lightweight transformer fusion, and bbox head.

        Args:
            siglip_model (nn.Module): Your Siglip2Encoders class with encode_text/image.
            fusion_transformer (nn.Module): Transformer encoder for fusion.
            bbox_head (nn.Module): MLP-based bbox regression head.
            criterion (nn.Module): RefExpLoss module combining L1 and IoU.
        """
        super().__init__()
        self.siglip = siglip_model
        self.fusion = fusion_transformer
        self.head = bbox_head
        self.criterion = criterion

        for param in self.siglip.parameters():
            param.requires_grad = False

    def forward(self, image, text, target_boxes=None, return_loss=True):
        """
        Args:
            image (PIL.Image or Tensor): Input image(s)
            text (str or List[str]): Input text(s)
            target_boxes (Tensor, optional): Ground truth boxes (B, 4) in normalized format
            return_loss (bool): Whether to compute and return loss

        Returns:
            Dict: {
                'pred_boxes': Tensor (B, 4),
                'loss': Tensor (optional),
                'loss_dict': Dict (optional)
            }
        """
        # Step A: SIGLIP encodings
        text_seq = self.siglip.encode_text(text, return_pooled=True)  # (B, D)
        image_seq = self.siglip.encode_image(image,
                                             return_pooled=True)  # (B, D)

        # Step B: Combine & fuse
        combined_seq = torch.cat([text_seq, image_seq], dim=1)
        fused_seq = self.fusion(combined_seq)

        # Step C: Predict bbox
        pred_boxes = self.head(fused_seq)  # (B, 4), in [0, 1]

        result = {"pred_boxes": pred_boxes}

        # Step D: Compute loss if target boxes are given
        if return_loss and target_boxes is not None:
            loss, loss_dict = self.criterion(
                pred_boxes,
                target_boxes,
            )
            result.update({"loss": loss, "loss_dict": loss_dict})

        return result
