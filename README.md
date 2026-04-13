# cell-segmentation-classification-bbbc038

Cell Segmentation & Classification using Deep Learning

Overview
This project presents an end-to-end pipeline for:
- Cell Segmentation (U-Net, DeepLabV3+)
- Cell Extraction
- Cell Classification (CNN)

Dataset: BBBC038 (Broad Bioimage Benchmark Collection)

---

Models Used
| Task | Model |
|------|------|
| Segmentation | U-Net (ResNet18, ResNet34), DeepLabV3+ |
| Classification | CNN |

---

Results

### Segmentation Performance
| Model | Dice | IoU |
|------|------|------|
| U-Net (ResNet18) | **0.8680** | **0.7691** |
| U-Net (ResNet34) | 0.8188 | 0.6964 |
| DeepLabV3+ | 0.8495 | 0.7400 |

Classification Performance
- Accuracy: **91.2%**
- Precision: **95.0%**
- Recall: **93.6%**
- F1 Score: **94.3%**

---

Ablation Study
| Experiment | Dice |
|----------|------|
| With Augmentation | 0.8188 |
| Without Augmentation | **0.8746** |

## ⚙️ Installation


pip install -r requirements.txt
