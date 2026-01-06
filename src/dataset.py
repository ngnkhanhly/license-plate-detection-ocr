src/dataset.py
# src/dataset.py
import torch
from torch.utils.data import Dataset
import albumentations as A
from albumentations.pytorch import ToTensorV2
import numpy as np
import cv2

class LicensePlateDataset(Dataset):
    """A thin wrapper around the HuggingFace dataset or custom list of dicts.

    Expected self.data entry format (as the HF dataset provides):
    {
        'image': <path or numpy array or PIL>,
        'image_id': int,
        'objects': {
            'bbox': [[xmin, ymin, width, height]],
            'area': float,
            'label': ['license_plate']
        }
    }
    """

    def __init__(self, data, transform=None):
        self.transform = transform
        self.data = data

    def __len__(self):
        return len(self.data)

    def _load_image(self, image):
        # image might be a path, numpy array or PIL
        if isinstance(image, str):
            img = cv2.imread(image)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            return img
        try:
            # assume numpy or PIL convertible
            arr = np.array(image)
            if arr.ndim == 2:
                arr = np.stack([arr]*3, axis=-1)
            return arr
        except Exception:
            raise RuntimeError('Unsupported image input type')

    def __getitem__(self, index):
        item = self.data[index]
        image = self._load_image(item['image'])

        # Convert bbox format to pascal_voc: [xmin, ymin, xmax, ymax]
        xmin = item['objects']['bbox'][0][0]
        ymin = item['objects']['bbox'][0][1]
        w = item['objects']['bbox'][0][2]
        h = item['objects']['bbox'][0][3]
        xmax = xmin + w
        ymax = ymin + h
        bbox = [xmin, ymin, xmax, ymax]

        labels = [1]  # single positive class (license_plate)

        if isinstance(self.transform, A.Compose):
            transformed = self.transform(image=image, bboxes=[bbox], labels=labels)
            image = transformed['image']
            bboxes = transformed['bboxes']
            labels = transformed['labels']
            boxes = torch.as_tensor(bboxes, dtype=torch.float32)
            labels = torch.as_tensor(labels, dtype=torch.int64)
        else:
            # Expect transform to be a torchvision transform or None
            boxes = torch.as_tensor([bbox], dtype=torch.float32)
            labels = torch.as_tensor([1], dtype=torch.int64)
            if self.transform is not None:
                # convert numpy to PIL if torchvision transforms expect PIL
                from torchvision.transforms import ToPILImage
                pil = ToPILImage()
                image = pil(torch.from_numpy(image).permute(2,0,1).float()/255.0)
                image = self.transform(image)

        target = {
            'boxes': boxes,
            'labels': labels,
            'image_id': torch.tensor(item.get('image_id', index)),
            'area': torch.as_tensor(item['objects'].get('area', (xmax-xmin)*(ymax-ymin)), dtype=torch.float32),
            'iscrowd': torch.zeros((len(labels),), dtype=torch.int64)
        }

        return image, target


def collate_fn(batch):
    return tuple(zip(*batch))
