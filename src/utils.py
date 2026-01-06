# src/utils.py
from PIL import Image, ImageDraw
import torch


def visualize_prediction(image_tensor, target, prediction, save_path=None):
    """Draw GT (green) and prediction (red) boxes on the image tensor and optionally save."""
    from torchvision.transforms import ToPILImage
    pil = ToPILImage()
    img = pil(image_tensor.cpu())
    draw = ImageDraw.Draw(img)

    # Ground truth
    for box in target['boxes'].cpu().numpy():
        draw.rectangle(list(box), outline='green', width=2)

    # Predictions: show top-5
    if len(prediction.get('boxes', [])) > 0:
        boxes = prediction['boxes'].cpu().numpy()
        scores = prediction.get('scores', torch.ones(len(boxes))).cpu().numpy()
        for box, score in zip(boxes[:5], scores[:5]):
            draw.rectangle(list(box), outline='red', width=2)
            draw.text((box[0], box[1]), f"{score:.2f}")

    if save_path:
        img.save(save_path)
    return img
