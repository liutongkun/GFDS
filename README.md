# GFDS
Exploring Few-Shot Defect Segmentation in General Industrial Scenarios with Metric Learning and Vision Foundation Models
## Preparation
rubber ring datasets: https://drive.google.com/file/d/1iFPSNfG_w8B8IntCiVFIFiIP9qWhdpRQ/view?usp=sharing

pretrained models: https://drive.google.com/file/d/1mTFBYodaZZeltezIgWERtGtuQT8cKSbM/view?usp=drive_link

FastSAM: https://github.com/CASIA-IVA-Lab/FastSAM

The publicly available dataset we used can be downloaded from the following link. For DAGM, we have re-annotated it, see'DAGM_finelabel.zip'.

    https://www.mvtec.com/company/research/datasets/mvtec-ad
    
    https://github.com/amazon-science/spot-diff
    
    https://www.dagm.de/the-german-association-for-pattern-recognition
    
    https://github.com/hmyao22/PSP-DS
    
    https://github.com/jianzhang96/MSD
    
    https://github.com/bbbbby-99/TGRNet-Surface-Defect-Segmentation

## Test
Run main.py, change all the involved path to your own path.

The structure of our file for the dataloader.py is as follows:

```
Dataset/
│
├── Large/
│   ├── images/
│   │   ├── bubble/
│   │   ├── wear/
│   │   ├── pit/
│   │   └── pressure/
│   └── ground_truth/
│       ├── bubble/
│       ├── wear/
│       ├── pit/
│       └── pressure/
│
├── Small/
│   ├── images/
│   │   └── ...
│   └── ground_truth/
│       └── ...
│
├── Side/
│   ├── images/
│   │   └── ...
│   └── ground_truth/
│       └── ...
```
## Acknowledgement:
we mainly use the code from https://github.com/fanq15/SSP, https://github.com/nelson1425/EfficientAD, https://github.com/CASIA-IVA-Lab/FastSAM. Thanks for their great work.

