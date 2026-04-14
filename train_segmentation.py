
import os, cv2, numpy as np, torch
import torch.nn as nn
import matplotlib.pyplot as plt

from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
import albumentations as A
import segmentation_models_pytorch as smp


# =========================
# 🔹 DATASET
# =========================
class SegDataset(Dataset):
    def __init__(self, folders, tf=None):
        self.folders = folders
        self.tf = tf

    def __len__(self):
        return len(self.folders)

    def __getitem__(self, idx):
        folder = self.folders[idx]

        img_path = os.path.join(folder,"images")
        mask_path = os.path.join(folder,"masks")

        img = cv2.imread(os.path.join(img_path, os.listdir(img_path)[0]))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        mask = None
        for f in os.listdir(mask_path):
            m = cv2.imread(os.path.join(mask_path,f),0)
            mask = m if mask is None else np.maximum(mask,m)

        mask = (mask > 0).astype("float32")

        if self.tf:
            aug = self.tf(image=img, mask=mask)
            img, mask = aug["image"], aug["mask"]

        img = torch.tensor(img/255.).permute(2,0,1).float()
        mask = torch.tensor(mask).unsqueeze(0).float()

        return img, mask


# =========================
# 🔹 METRICS
# =========================
def dice_score(p,y):
    p = (torch.sigmoid(p)>0.5).float()
    return (2*(p*y).sum()+1e-6)/((p+y).sum()+1e-6)

def iou_score(p,y):
    p = (torch.sigmoid(p)>0.5).float()
    inter = (p*y).sum()
    union = p.sum()+y.sum()-inter
    return (inter+1e-6)/(union+1e-6)


# =========================
# 🔹 TRAIN
# =========================
def train_model(model, train_loader, loss_fn, device, name, epochs=5):
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)

    for ep in range(epochs):
        model.train()
        total_loss = 0

        for x,y in train_loader:
            x,y = x.to(device), y.to(device)
            p = model(x)

            loss = loss_fn(p,y)

            opt.zero_grad()
            loss.backward()
            opt.step()

            total_loss += loss.item()

        print(f"{name} Epoch {ep} Loss {total_loss:.4f}")

    torch.save(model.state_dict(), f"{name}.pth")
    return model


# =========================
# 🔹 EVAL
# =========================
def eval_model(model, val_loader, device):
    model.eval()
    d,i,n = 0,0,0

    with torch.no_grad():
        for x,y in val_loader:
            x,y = x.to(device), y.to(device)
            p = model(x)

            d += dice_score(p,y).item()
            i += iou_score(p,y).item()
            n += 1

    return d/n, i/n


# =========================
# 🔹 MAIN
# =========================
def main():

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Device:", device)

    folders = [
        f for f in os.listdir(".")
        if os.path.isdir(f)
        and os.path.exists(os.path.join(f,"images"))
        and os.path.exists(os.path.join(f,"masks"))
    ]

    train_folders, val_folders = train_test_split(folders, test_size=0.2, random_state=42)

    train_tf = A.Compose([
        A.Resize(128,128),
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5),
    ])

    val_tf = A.Compose([
        A.Resize(128,128),
    ])

    train_loader = DataLoader(SegDataset(train_folders, train_tf), batch_size=8, shuffle=True)
    val_loader   = DataLoader(SegDataset(val_folders, val_tf), batch_size=8)

    # Models
    baseline = smp.Unet("resnet18", encoder_weights="imagenet", classes=1).to(device)
    improved = smp.Unet("resnet34", encoder_weights="imagenet", classes=1).to(device)
    deeplab  = smp.DeepLabV3Plus("resnet34", encoder_weights="imagenet", classes=1).to(device)

    # Loss
    bce = nn.BCEWithLogitsLoss()
    dice = smp.losses.DiceLoss(mode='binary')

    def loss_baseline(p,y): return bce(p,y)
    def loss_improved(p,y): return 0.5*bce(p,y) + 0.5*dice(p,y)

    # Train
    baseline = train_model(baseline, train_loader, loss_baseline, device, "baseline")
    improved = train_model(improved, train_loader, loss_improved, device, "improved")
    deeplab  = train_model(deeplab,  train_loader, loss_improved, device, "deeplab")

    # Evaluate
    print("\n===== RESULTS =====")
    for name, model in [("Baseline",baseline),("Improved",improved),("DeepLab",deeplab)]:
        d,i = eval_model(model, val_loader, device)
        print(f"{name} Dice:{d:.4f} IoU:{i:.4f}")


if __name__ == "__main__":
    main()
