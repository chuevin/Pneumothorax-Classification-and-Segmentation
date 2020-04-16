# Pneumothorax-Classification-and-Segmentation

## Introduction

Pneumothorax is the medical term used for a collapsed lung. This medical condition occurs when air leaks into the space between the lung lobe and pulmonary wall. It is a rare condition, there are less than a million reported cases per year in India. Although it is a rare condition and typically resolves within weeks, emergency care is needed. The most common symptoms for a pneumothorax are sudden chest pain and shortness of breath. On some occasions, a collapsed lung can be a life-threatening event.

Pneumothorax is usually diagnosed by a radiologist on a chest x-ray, and can sometimes be very difficult to confirm. An accurate AI algorithm to detect pneumothorax would be useful in a lot of clinical scenarios. AI could be used to triage chest radiographs for priority interpretation, or to provide a more confident diagnosis for non-radiologists.

In this project we present a Computed Aided Diagnosis(CAD) system to classify and if present, segment pneumothorax from a set of chest radiographic images. This problem can be solved by breaking the problem down into smaller subproblems, then these smaller subproblems can be solved easily by relating them to other problems that have been studied in great detail in other domains.

First, the lung boundary is detected from the given image. This is done using a Fully Convolutional Model which is based on the U-Net architecture. Next, the lung region is cropped out and sent for classification. We have proposed two approaches for classification, the first approach uses features extracted from Uniform Local Binary Patterns(ULBP) and classifies using a Support Vector Machine(SVM). 

Next, pneumothorax segmentation is performed by a modified U-Net architecture. U-Net architecture have already been used in biomedical imaging.

We achieve state-of-the-art results in classification and segmentation making our model quite robust and accurate. Our complete classification and segmentation pipeline is very efficient and can be easily deployed if required.

## Dataset

For experimentation,  the dataset is taken from [Kaggle](https://www.kaggle.com/c/siim-acr-pneumothorax-segmentation/overview), the dataset has over 13,000  Chest X-ray RGB images in Digital Imaging and Communications in Medicine (DICOM) format. DICOM is the standard for the communication and management of medical imaging information and related data. DICOM is most commonly used for storing and transmitting medical images enabling the integration of medical imaging devices such as scanners, servers, workstations, printers, network hardware, and picture archiving and communication systems (PACS) from multiple manufacturers. It has been widely adopted by hospitals and is making inroads into smaller applications like dentists' and doctors' offices.
The dimension of the image in the dataset is 1024x1024.


<img src="images/dataset_imgs.png" >

## Prerequisites

* Python (3.6)
* Keras (2.3.0)
* Tensorflow (2.1)
* NumPy (1.14.3)
* Sklearn (0.22.2)
* Albumentations (0.4.5)
* pandas (0.22.0)
* tqdm (4.19.8)
* opencv-python (3.4.2)
* pydicom (1.4.2)

## Augmentations

Used Albumentations library for image augmentation
```
train_augment = Compose([
    HorizontalFlip(p = 0.5),
    ShiftScaleRotate(p = 0.5),
    ElasticTransform(alpha=120, sigma=120 * 0.05, alpha_affine=120 * 0.03,p = 0.5),
    OneOf([
        RandomContrast(),
        RandomGamma(),
        RandomBrightness(),
         ], p=0.3),
    OneOf([
        RandomSizedCrop(min_max_height=(176, 256), height=256, width=256,p=0.25),
        CLAHE(clip_limit=2.0, tile_grid_size=(32,32),p = 0.4),
         ], p=0.5),
    ToFloat()
])
```

The above augmentations are applied to dataset as:
<img src="images/pneumothorax.png" >

## Results Table

The following table shows performance of different ML models on binary classification task:

| Architecture | Classification Accuracy |
| --------------- | --------------- |
| DenseNet 169 | **91.23%** |
| ResNet50 | 86.52% |
| ULBP + SVM without Lung Segmentation | 54.79% |
| ULBP + SVM with Lung Segmentation | 78.20% |

The following table shows performance of different Unet-based models on metrics such as IoU and Dice Coefficent:

| Architecture | Model IoU | Model Dice Coefficient |
| --------------- | --------------- | --------------- |
| SE-ResNet34Unet | 0.7566 | 0.7737 |
| SE-ResNeXt50UNet | 0.7877 | 0.8223 |
| XceptionUNet | 0.7629 | 0.8157 |
| EfficientNetUNet | **0.8106** | **0.8420** |

## License

This repository is licensed under the terms of the MIT license.
