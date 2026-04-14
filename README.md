# 🔬 Cell Segmentation & Classification

## 📌 Overview
End-to-end pipeline for:
- Cell Segmentation (U-Net, DeepLabV3+)
- Cell Extraction
- Cell Classification (CNN)

## 📊 Results

### Segmentation
| Model | Dice | IoU |
|------|------|------|
| U-Net (ResNet18) | 0.8680 | 0.7691 |
| U-Net (ResNet34) | 0.8188 | 0.6964 |
| DeepLabV3+ | 0.8495 | 0.7400 |

### Classification
- Accuracy: 91.2%
- F1 Score: 94.3%

## 🚀 Run
pip install -r requirements.txt

python train_segmentation.py
python train_classification.py

## 👨‍💻 Author
Zia Ul Rehman Zafar
