import torchvision
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.rpn import AnchorGenerator, RPNHead

import torch


def get_model(num_classes=2, nms_thresh=0.3, weights=None):
    """Create Faster R-CNN with ResNet50-FPN backbone.

    `weights` can be a string like 'COCO_V1' or None. For torchvision>=0.13
    you can pass torchvision.models.detection.FasterRCNN_ResNet50_FPN_Weights.COCO_V1
    """
    # Use torchvision helper (handles weights enums if available)
    try:
        if weights is None:
            model = torchvision.models.detection.fasterrcnn_resnet50_fpn(weights=None)
        else:
            # torchvision >= 0.13 provides weights enums
            model = torchvision.models.detection.fasterrcnn_resnet50_fpn(weights=weights)
    except Exception:
        model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=(weights is not None))

    # Replace the classifier head
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)

    # Custom anchors tuned for wide license plates
    anchor_sizes = ((32,), (64,), (128,), (256,), (512,))
    aspect_ratios = ((0.25, 0.5, 1.0, 2.0, 4.0),) * len(anchor_sizes)
    anchor_generator = AnchorGenerator(sizes=anchor_sizes, aspect_ratios=aspect_ratios)
    model.rpn.anchor_generator = anchor_generator

    out_channels = model.backbone.out_channels
    num_anchors = len(aspect_ratios[0]) * len(anchor_sizes[0])
    model.rpn.head = RPNHead(out_channels, num_anchors)

    model.roi_heads.nms_thresh = nms_thresh
    model.roi_heads.score_thresh = 0.05

    return model


# small helper for IoU used in evaluation

def box_iou(boxes1, boxes2):
    if boxes1.numel() == 0 or boxes2.numel() == 0:
        return torch.zeros((boxes1.shape[0], boxes2.shape[0]))
    area1 = (boxes1[:, 2] - boxes1[:, 0]) * (boxes1[:, 3] - boxes1[:, 1])
    area2 = (boxes2[:, 2] - boxes2[:, 0]) * (boxes2[:, 3] - boxes2[:, 1])

    lt = torch.max(boxes1[:, None, :2], boxes2[:, :2])
    rb = torch.min(boxes1[:, None, 2:], boxes2[:, 2:])
    wh = (rb - lt).clamp(min=0)
    inter = wh[:, :, 0] * wh[:, :, 1]
    union = area1[:, None] + area2 - inter
    return inter / union
