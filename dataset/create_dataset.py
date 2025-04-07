from dataset.h5_dataset import ImageEmbDataset,TextEmbDataset,ImageTextEmbDataset
from dataset.img_dataset import MIMIC_Img_Dataset
from dataset.img_text_dataset import MIMIC_Img_text_Dataset
import torchvision
import albumentations as A
from PIL import Image
import numpy as np
import math
class MedAugmentTransform:
    def __init__(self, level=1):
        self.level = level
        self.transform = A.Compose([
            A.ColorJitter(brightness=0.04 * level, contrast=0, saturation=0, hue=0, p=0.2 * level),
            A.ColorJitter(brightness=0, contrast=0.04 * level, saturation=0, hue=0, p=0.2 * level),
            A.Posterize(num_bits=max(1, math.floor(8 - 0.8 * level)), p=0.2 * level),
            A.Sharpen(alpha=(0.04 * level, 0.1 * level), lightness=(1, 1), p=0.2 * level),
            A.GaussianBlur(blur_limit=(3, self._odd(3 + 0.8 * level)), p=0.2 * level),
            A.GaussNoise(var_limit=(2 * level, 10 * level), mean=0, per_channel=True, p=0.2 * level),
            A.Rotate(limit=4 * level, interpolation=1, border_mode=0, value=0, p=0.2 * level),
            A.HorizontalFlip(p=0.2 * level),
            A.VerticalFlip(p=0.2 * level),
            A.Affine(scale=(1 - 0.04 * level, 1 + 0.04 * level), translate_percent=None, translate_px=None, rotate=None,
                 shear=None, interpolation=1, mask_interpolation=0, cval=0, cval_mask=0, mode=0, fit_output=False,
                 keep_ratio=True, p=0.2 * level),
            A.Affine(scale=None, translate_percent=None, translate_px=None, rotate=None,
                    shear={'x': (0, 2 * level), 'y': (0, 0)}
                    , interpolation=1, mask_interpolation=0, cval=0, cval_mask=0, mode=0, fit_output=False,
                    keep_ratio=True, p=0.2 * level),  # x
            A.Affine(scale=None, translate_percent=None, translate_px=None, rotate=None,
                    shear={'x': (0, 0), 'y': (0, 2 * level)}
                    , interpolation=1, mask_interpolation=0, cval=0, cval_mask=0, mode=0, fit_output=False,
                    keep_ratio=True, p=0.2 * level),
            A.Affine(scale=None, translate_percent={'x': (0, 0.02 * level), 'y': (0, 0)}, translate_px=None, rotate=None,
                    shear=None, interpolation=1, mask_interpolation=0, cval=0, cval_mask=0, mode=0, fit_output=False,
                    keep_ratio=True, p=0.2 * level),
            A.Affine(scale=None, translate_percent={'x': (0, 0), 'y': (0, 0.02 * level)}, translate_px=None, rotate=None,
                    shear=None, interpolation=1, mask_interpolation=0, cval=0, cval_mask=0, mode=0, fit_output=False,
                    keep_ratio=True, p=0.2 * level)
    ])
        
    def _odd(self, x):
        # 保证 blur_limit 为奇数
        x = int(round(x))
        return x if x % 2 == 1 else x+1

    def __call__(self, img):
        img_np = np.array(img)
        # Albumentations 期望的图像格式是 uint8
        augmented = self.transform(image=img_np)
        aug_img = augmented['image']
        # 转换回 PIL Image
        return Image.fromarray(aug_img)





def create_dataset(name, **kwargs):
    if name == 'ImageEmbDataset':
        path_dict = kwargs.get('h5_file_path')
        return ImageEmbDataset(path_dict['train']),ImageEmbDataset(path_dict['val']),ImageEmbDataset(path_dict['test'])
    elif name == 'TextEmbDataset':
        path_dict = kwargs.get('h5_file_path')
        return TextEmbDataset(path_dict['train']),TextEmbDataset(path_dict['val']),TextEmbDataset(path_dict['test'])
    elif name == 'ImageTextEmbDataset':
        path_dict = kwargs.get('h5_file_path')
        return ImageTextEmbDataset(path_dict['train']),ImageTextEmbDataset(path_dict['val']),ImageTextEmbDataset(path_dict['test'])
    elif name == 'MIMIC_Img_Dataset':
        path_dict = kwargs.get('h5_file_path')
        train_dataset = MIMIC_Img_Dataset(path_dict['train'],kwargs.get('img_h5_path')['train'],transforms = torchvision.transforms.Compose([
            torchvision.transforms.RandomResizedCrop(224, scale=(0.9, 1.0)),
            # torchvision.transforms.RandomHorizontalFlip(),
            # torchvision.transforms.RandomRotation(15),
            MedAugmentTransform(level=1),
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))
        ]))
        val_dataset = MIMIC_Img_Dataset(path_dict['val'],kwargs.get('img_h5_path')['val'])
        test_dataset = MIMIC_Img_Dataset(path_dict['test'],kwargs.get('img_h5_path')['test'])
        return train_dataset,val_dataset,test_dataset
    elif name == 'MIMIC_Img_text_Dataset':
        path_dict = kwargs.get('h5_file_path')
        train_dataset = MIMIC_Img_text_Dataset(path_dict['train'],kwargs.get('imgpath'),transforms = torchvision.transforms.Compose([
            torchvision.transforms.RandomResizedCrop(224, scale=(0.9, 1.0)),
            # torchvision.transforms.RandomHorizontalFlip(),
            # torchvision.transforms.RandomRotation(15),
            MedAugmentTransform(level=1),
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))
        ]))
        val_dataset = MIMIC_Img_text_Dataset(path_dict['val'],kwargs.get('imgpath'))
        test_dataset = MIMIC_Img_text_Dataset(path_dict['test'],kwargs.get('imgpath'))
        return train_dataset,val_dataset,test_dataset
    else:
        raise ValueError(f"未知数据集类型: {name}")
