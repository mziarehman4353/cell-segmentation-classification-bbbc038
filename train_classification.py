
import os, cv2, numpy as np, torch
import torch.nn as nn
import seaborn as sns
import matplotlib.pyplot as plt

from torch.utils.data import DataLoader
from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score
from skimage.measure import label, regionprops


# =========================
# 🔹 CELL EXTRACTION
# =========================
def improved_label(r):
    return 0 if r.area<200 and r.eccentricity<0.8 else 1

def extract_cells(img, mask):
    labeled = label(mask)
    cells, labels = [], []

    for r in regionprops(labeled):
        if r.area<50: continue
        y1,x1,y2,x2 = r.bbox
        c = cv2.resize(img[y1:y2,x1:x2],(64,64))
        cells.append(c)
        labels.append(improved_label(r))

    return cells, labels


# =========================
# 🔹 LOAD DATA
# =========================
def load_data():

    folders = [
        f for f in os.listdir(".")
        if os.path.isdir(f)
        and os.path.exists(os.path.join(f,"images"))
        and os.path.exists(os.path.join(f,"masks"))
    ]

    all_cells, all_labels = [], []

    for folder in folders:
        img = cv2.imread(os.path.join(folder,"images",os.listdir(os.path.join(folder,"images"))[0]))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        mask=None
        for f in os.listdir(os.path.join(folder,"masks")):
            m = cv2.imread(os.path.join(folder,"masks",f),0)
            mask = m if mask is None else np.maximum(mask,m)

        c,l = extract_cells(img,mask)
        all_cells += c
        all_labels += l

    X = torch.tensor(np.array(all_cells)/255.).permute(0,3,1,2).float()
    y = torch.tensor(all_labels).long()

    return DataLoader(list(zip(X,y)), batch_size=32, shuffle=True)


# =========================
# 🔹 MODEL
# =========================
class Classifier(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(3,16,3,padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(16,32,3,padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Flatten(),
            nn.Linear(32*16*16,64),
            nn.ReLU(),
            nn.Linear(64,2)
        )

    def forward(self,x):
        return self.net(x)


# =========================
# 🔹 TRAIN
# =========================
def main():

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Device:", device)

    loader = load_data()

    model = Classifier().to(device)
    opt = torch.optim.Adam(model.parameters(),1e-3)
    loss_fn = nn.CrossEntropyLoss()

    for ep in range(5):
        for x,y in loader:
            x,y = x.to(device), y.to(device)

            p = model(x)
            loss = loss_fn(p,y)

            opt.zero_grad()
            loss.backward()
            opt.step()

        print(f"Epoch {ep} Loss {loss.item():.4f}")

    torch.save(model.state_dict(), "classifier.pth")

    # =========================
    # 🔹 EVALUATION
    # =========================
    all_preds, all_gt = [], []

    with torch.no_grad():
        for x,y in loader:
            x = x.to(device)
            preds = model(x).argmax(1).cpu().numpy()
            all_preds.extend(preds)
            all_gt.extend(y.numpy())

    print("Accuracy:", np.mean(np.array(all_preds)==np.array(all_gt)))
    print("Precision:", precision_score(all_gt, all_preds))
    print("Recall:", recall_score(all_gt, all_preds))
    print("F1:", f1_score(all_gt, all_preds))

    cm = confusion_matrix(all_gt, all_preds)

    sns.heatmap(cm, annot=True, fmt='d')
    plt.title("Confusion Matrix")
    plt.savefig("confusion_matrix.png")
    plt.show()


if __name__ == "__main__":
    main()
