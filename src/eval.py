# src/eval.py
import argparse
from pathlib import Path
import torch
from torch.utils.data import DataLoader
import torchvision.transforms as T
from datasets import load_dataset
from torchmetrics.detection.mean_ap import MeanAveragePrecision

from src.dataset import LicensePlateDataset, collate_fn
from src.model import get_model, box_iou
from src.utils import visualize_prediction


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--weights', type=str, required=True, help='path to saved model state_dict')
    p.add_argument('--use-hf', action='store_true')
    p.add_argument('--local-data', type=str, default=None)
    p.add_argument('--batch', type=int, default=4)
    p.add_argument('--no-cuda', action='store_true')
    return p.parse_args()


def main():
    args = parse_args()
    device = 'cuda' if torch.cuda.is_available() and not args.no_cuda else 'cpu'

    if args.use_hf:
        ds = load_dataset('keremberke/license-plate-object-detection', 'full')
        test_ds = ds['test']
    else:
        import json
        with open(args.local_data, 'r') as f:
            data = json.load(f)
        test_ds = data['test']

    test_dataset = LicensePlateDataset(test_ds, transform=T.ToTensor())
    test_loader = DataLoader(test_dataset, batch_size=args.batch, shuffle=False, collate_fn=collate_fn)

    num_classes = 2
    model = get_model(num_classes=num_classes)
    model.load_state_dict(torch.load(args.weights, map_location=device))
    model.to(device)
    model.eval()

    map_metric = MeanAveragePrecision()

    all_preds = []
    all_targets = []

    with torch.no_grad():
        for images, targets in test_loader:
            images = [img.to(device) for img in images]
            targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

            predictions = model(images)
            preds_cpu = [{k: v.cpu() for k, v in p.items()} for p in predictions]
            targets_cpu = [{k: v.cpu() for k, v in t.items()} for t in targets]
            map_metric.update(preds_cpu, targets_cpu)

    map_results = map_metric.compute()
    print(map_results)


if __name__ == '__main__':
    main()
