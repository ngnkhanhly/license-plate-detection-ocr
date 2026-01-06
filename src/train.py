# src/train.py
import argparse
import os
from pathlib import Path
import torch
from torch.utils.data import DataLoader
import torchvision.transforms as T
import albumentations as A
from albumentations.pytorch import ToTensorV2
from datasets import load_dataset

from src.dataset import LicensePlateDataset, collate_fn
from src.model import get_model


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--epochs', type=int, default=6)
    p.add_argument('--batch', type=int, default=4)
    p.add_argument('--lr', type=float, default=0.005)
    p.add_argument('--out', type=str, default='models')
    p.add_argument('--weights', type=str, default='COCO_V1')
    p.add_argument('--no-cuda', action='store_true')
    p.add_argument('--use-hf', action='store_true', help='Load dataset from HuggingFace (keremberke/license-plate-object-detection)')
    p.add_argument('--local-data', type=str, default=None, help='Path to local dataset if not using HF')
    return p.parse_args()


def main():
    args = parse_args()
    device = 'cuda' if torch.cuda.is_available() and not args.no_cuda else 'cpu'

    # dataset transforms
    train_transform = A.Compose([
        A.HorizontalFlip(p=0.5),
        A.RandomBrightnessContrast(p=0.3),
        A.ShiftScaleRotate(shift_limit=0.05, scale_limit=0.1, rotate_limit=15, p=0.5),
        A.RandomCrop(width=512, height=512, p=0.3),
        ToTensorV2()
    ], bbox_params=A.BboxParams(format='pascal_voc', label_fields=['labels']))

    # Load dataset
    if args.use_hf:
        ds = load_dataset('keremberke/license-plate-object-detection', 'full')
        train_ds = ds['train']
        val_ds = ds['validation']
    else:
        assert args.local_data is not None, 'Either --use-hf or --local-data must be provided.'
        # If user provides a local dataset as a list-of-dicts saved as .pt or .json, load it here.
        import json
        with open(args.local_data, 'r') as f:
            data = json.load(f)
        # Expect data has 'train' and 'validation' lists
        train_ds = data['train']
        val_ds = data['validation']

    train_dataset = LicensePlateDataset(train_ds, transform=train_transform)
    val_dataset = LicensePlateDataset(val_ds, transform=T.ToTensor())

    train_loader = DataLoader(train_dataset, batch_size=args.batch, shuffle=True, collate_fn=collate_fn)
    val_loader = DataLoader(val_dataset, batch_size=args.batch, shuffle=False, collate_fn=collate_fn)

    num_classes = 2
    # If user specified string weight name, attempt to map to torchvision enum
    weights = None
    try:
        if args.weights == 'COCO_V1':
            # newer torchvision uses enum, attempt to import
            from torchvision.models.detection import FasterRCNN_ResNet50_FPN_Weights
            weights = FasterRCNN_ResNet50_FPN_Weights.COCO_V1
    except Exception:
        weights = None

    model = get_model(num_classes=num_classes, nms_thresh=0.3, weights=weights)
    model.to(device)

    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.SGD(params, lr=args.lr, momentum=0.9, weight_decay=0.0005)
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=3, gamma=0.1)

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    model.train()
    for epoch in range(args.epochs):
        running_loss = 0.0
        for batch_idx, (images, targets) in enumerate(train_loader):
            images = list(img.to(device) for img in images)
            targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

            loss_dict = model(images, targets)
            losses = sum(loss for loss in loss_dict.values())

            optimizer.zero_grad()
            losses.backward()
            optimizer.step()

            running_loss += losses.item()
            if batch_idx % 50 == 0:
                print(f"Epoch {epoch} Batch {batch_idx} Loss {losses.item():.4f}")

        print(f"Epoch {epoch} Train loss: {running_loss/len(train_loader):.4f}")

        # simple validation loss computation
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for images, targets in val_loader:
                images = list(img.to(device) for img in images)
                targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
                val_loss_dict = model(images, targets)
                val_loss += sum(val_loss_dict.values()).item()
        val_loss = val_loss / max(1, len(val_loader))
        print(f"Epoch {epoch} Validation loss: {val_loss:.4f}")

        # save model
        torch.save(model.state_dict(), out_dir / f"frcnn_epoch{epoch}.pth")

        model.train()
        if lr_scheduler is not None:
            lr_scheduler.step()

    print('Training finished')


if __name__ == '__main__':
    main()
