import cv2
from albumentations import (HorizontalFlip, IAAPerspective, ShiftScaleRotate, CLAHE, RandomRotate90,
    Transpose, ShiftScaleRotate, Blur, OpticalDistortion, GridDistortion, ElasticTransform, HueSaturationValue,
    IAAAdditiveGaussianNoise, GaussNoise, MotionBlur, MedianBlur, IAAPiecewiseAffine,
    IAASharpen, IAAEmboss, RandomBrightnessContrast, Flip, OneOf, Compose, RandomBrightness, RandomContrast, RandomGamma,
    Normalize, Resize)
from albumentations.torch import ToTensor

def weak_aug(p,img_size, mean, std):
    return Compose([
        HorizontalFlip(p=0.5),
        OneOf([
            GaussNoise(),
            Blur(5)
        ], p=0.5),
        ShiftScaleRotate(shift_limit=0.0225, scale_limit=(-0.15,0.15), rotate_limit=20, p=1.0, border_mode=cv2.BORDER_REPLICATE),
        OneOf([
                RandomBrightnessContrast(0.2, 0.05),
                RandomGamma(gamma_limit=(70,130)),
                CLAHE(clip_limit=0.1)
            ], p=1),
        Normalize(mean, std, max_pixel_value=255.0, p=1.0),
        Resize(img_size,img_size)
        ], 
        p=p)

def val_aug(p,img_size, mean, std):
    return Compose([
            Normalize(mean, std, max_pixel_value=255.0, p=1.0),
            Resize(img_size,img_size)
        ], 
        p=p)