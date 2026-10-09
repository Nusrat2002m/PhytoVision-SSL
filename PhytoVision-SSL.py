!pip install -q kagglehub torch torchvision scikit-learn seaborn matplotlib pandas tqdm

import os, random, time, copy
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, Subset
import torchvision
from torchvision import transforms, models
from sklearn.metrics import (accuracy_score, precision_recall_fscore_support,
                              confusion_matrix, classification_report)
from sklearn.manifold import TSNE

plt.rcParams['figure.dpi'] = 400
sns.set_style("whitegrid")

SEED = 42
random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Device:", device)

# --------------------------
# 2. DOWNLOAD DATASET
# --------------------------
import kagglehub
path = kagglehub.dataset_download("vipoooool/new-plant-diseases-dataset")
print("Path to dataset files:", path)

# Locate the actual train/valid folders (structure nested inside path)
for root, dirs, files in os.walk(path):
    if 'train' in [d.lower() for d in dirs]:
        base_dir = root
        break
print("Base dir:", base_dir)

train_dir = os.path.join(base_dir, [d for d in os.listdir(base_dir) if d.lower()=='train'][0])
valid_dir = os.path.join(base_dir, [d for d in os.listdir(base_dir) if d.lower() in ('valid','validation')][0])

classes = sorted(os.listdir(train_dir))
num_classes = len(classes)
print(f"Found {num_classes} classes")

# --------------------------
# 3. EXPLORATORY DATA ANALYSIS + TABLE
# --------------------------
class_counts = {c: len(os.listdir(os.path.join(train_dir, c))) for c in classes}
df_counts = pd.DataFrame(list(class_counts.items()), columns=["Class", "Image_Count"]).sort_values("Image_Count", ascending=False)
print(df_counts.to_string(index=False))
df_counts.to_csv("class_distribution.csv", index=False)

# Table: dataset summary (paper-ready)
summary_table = pd.DataFrame({
    "Attribute": ["Total Classes", "Total Training Images", "Total Validation Images",
                  "Avg Images per Class", "Min Images per Class", "Max Images per Class"],
    "Value": [num_classes, sum(class_counts.values()),
              sum(len(os.listdir(os.path.join(valid_dir, c))) for c in classes),
              int(np.mean(list(class_counts.values()))),
              min(class_counts.values()), max(class_counts.values())]
})
print(summary_table.to_string(index=False))
summary_table.to_csv("dataset_summary_table.csv", index=False)

# Figure 1: Class distribution bar chart (top 15 for readability)
plt.figure(figsize=(10,6))
top15 = df_counts.head(15)
sns.barplot(data=top15, x="Image_Count", y="Class", palette="viridis")
plt.title("Figure 1: Top-15 Class Distribution in Training Set", fontsize=12)
plt.xlabel("Number of Images"); plt.ylabel("Class")
plt.tight_layout()
plt.savefig("fig1_class_distribution.png", dpi=400, bbox_inches='tight')
plt.show()

# Figure 2: Sample images grid
def show_sample_grid(n=9):
    fig, axes = plt.subplots(3, 3, figsize=(9,9))
    sample_classes = random.sample(classes, 9)
    for ax, cls in zip(axes.flatten(), sample_classes):
        img_dir = os.path.join(train_dir, cls)
        img_file = random.choice(os.listdir(img_dir))
        img = plt.imread(os.path.join(img_dir, img_file))
        ax.imshow(img); ax.set_title(cls, fontsize=8); ax.axis('off')
    plt.suptitle("Figure 2: Sample Leaf Images Across Classes", fontsize=12)
    plt.tight_layout()
    plt.savefig("fig2_sample_grid.png", dpi=400, bbox_inches='tight')
    plt.show()

show_sample_grid()

# --------------------------
# 4. DATA AUGMENTATION (for SSL contrastive views)
# --------------------------
IMG_SIZE = 224

simclr_transform = transforms.Compose([
    transforms.RandomResizedCrop(IMG_SIZE, scale=(0.5, 1.0)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomApply([transforms.ColorJitter(0.4,0.4,0.4,0.1)], p=0.8),
    transforms.RandomGrayscale(p=0.2),
    transforms.GaussianBlur(kernel_size=23),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225])
])

eval_transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225])
])

# --------------------------
# 5. SSL DATASET (returns two augmented views, no labels used)
# --------------------------
class ContrastiveDataset(Dataset):
    def __init__(self, root_dir, transform):
        self.dataset = torchvision.datasets.ImageFolder(root_dir)
        self.transform = transform
    def __len__(self):
        return len(self.dataset)
    def __getitem__(self, idx):
        img, _ = self.dataset[idx]   # label discarded -> self-supervised
        v1 = self.transform(img)
        v2 = self.transform(img)
        return v1, v2

# Use a subset for pretraining speed on Colab (adjust as needed)
PRETRAIN_SUBSET_SIZE = 20000
full_train_folder = torchvision.datasets.ImageFolder(train_dir)
indices = random.sample(range(len(full_train_folder)), min(PRETRAIN_SUBSET_SIZE, len(full_train_folder)))

ssl_dataset = ContrastiveDataset(train_dir, simclr_transform)
ssl_subset = Subset(ssl_dataset, indices)
ssl_loader = DataLoader(ssl_subset, batch_size=128, shuffle=True, num_workers=2, drop_last=True)

# --------------------------
# 6. SIMCLR MODEL: ResNet50 Encoder + Projection Head
# --------------------------
class SimCLR(nn.Module):
    def __init__(self, base_model='resnet50', proj_dim=128):
        super().__init__()
        backbone = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
        self.feature_dim = backbone.fc.in_features
        backbone.fc = nn.Identity()
        self.encoder = backbone
        self.projector = nn.Sequential(
            nn.Linear(self.feature_dim, self.feature_dim),
            nn.ReLU(inplace=True),
            nn.Linear(self.feature_dim, proj_dim)
        )
    def forward(self, x):
        h = self.encoder(x)
        z = self.projector(h)
        return h, F.normalize(z, dim=1)

def nt_xent_loss(z1, z2, temperature=0.5):
    batch_size = z1.size(0)
    z = torch.cat([z1, z2], dim=0)
    sim = torch.mm(z, z.t()) / temperature
    sim.fill_diagonal_(-1e9)
    pos_idx = torch.cat([torch.arange(batch_size, 2*batch_size), torch.arange(0, batch_size)]).to(z.device)
    loss = F.cross_entropy(sim, pos_idx)
    return loss

model = SimCLR().to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=3e-4, weight_decay=1e-6)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=20)

# --------------------------
# 7. SSL PRETRAINING LOOP
# --------------------------
EPOCHS_SSL = 20
ssl_losses = []

for epoch in range(EPOCHS_SSL):
    model.train()
    epoch_loss = 0
    for v1, v2 in tqdm(ssl_loader, desc=f"SSL Epoch {epoch+1}/{EPOCHS_SSL}"):
        v1, v2 = v1.to(device), v2.to(device)
        _, z1 = model(v1)
        _, z2 = model(v2)
        loss = nt_xent_loss(z1, z2)
        optimizer.zero_grad(); loss.backward(); optimizer.step()
        epoch_loss += loss.item()
    scheduler.step()
    avg_loss = epoch_loss / len(ssl_loader)
    ssl_losses.append(avg_loss)
    print(f"Epoch {epoch+1}: Contrastive Loss = {avg_loss:.4f}")

torch.save(model.encoder.state_dict(), "phytovision_ssl_encoder.pth")

# Figure 3: SSL pretraining loss curve
plt.figure(figsize=(7,5))
plt.plot(range(1, EPOCHS_SSL+1), ssl_losses, marker='o', color='darkgreen')
plt.title("Figure 3: Self-Supervised Contrastive Pretraining Loss")
plt.xlabel("Epoch"); plt.ylabel("NT-Xent Loss")
plt.tight_layout()
plt.savefig("fig3_ssl_loss_curve.png", dpi=400, bbox_inches='tight')
plt.show()

# --------------------------
# 8. FINE-TUNING (labelled subset, e.g., 15% of data)
# --------------------------
class LabelledDataset(Dataset):
    def __init__(self, root_dir, transform, fraction=0.15):
        full = torchvision.datasets.ImageFolder(root_dir)
        n = int(len(full) * fraction)
        idx = random.sample(range(len(full)), n)
        self.samples = [full.samples[i] for i in idx]
        self.transform = transform
        self.loader = full.loader
    def __len__(self):
        return len(self.samples)
    def __getitem__(self, idx):
        path, label = self.samples[idx]
        img = self.loader(path)
        return self.transform(img), label

train_ft_dataset = LabelledDataset(train_dir, eval_transform, fraction=0.15)
val_dataset = torchvision.datasets.ImageFolder(valid_dir, transform=eval_transform)

train_ft_loader = DataLoader(train_ft_dataset, batch_size=64, shuffle=True, num_workers=2)
val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False, num_workers=2)

class Classifier(nn.Module):
    def __init__(self, encoder, feature_dim, num_classes):
        super().__init__()
        self.encoder = encoder
        self.fc = nn.Linear(feature_dim, num_classes)
    def forward(self, x):
        with torch.no_grad():
            h = self.encoder(x)
        return self.fc(h)

clf_model = Classifier(model.encoder, model.feature_dim, num_classes).to(device)
for p in clf_model.encoder.parameters():
    p.requires_grad = False  # linear evaluation protocol

ft_optimizer = torch.optim.Adam(clf_model.fc.parameters(), lr=1e-3)
criterion = nn.CrossEntropyLoss()

EPOCHS_FT = 15
train_acc_hist, val_acc_hist = [], []

for epoch in range(EPOCHS_FT):
    clf_model.train()
    correct, total = 0, 0
    for imgs, labels in tqdm(train_ft_loader, desc=f"Fine-tune Epoch {epoch+1}/{EPOCHS_FT}"):
        imgs, labels = imgs.to(device), labels.to(device)
        out = clf_model(imgs)
        loss = criterion(out, labels)
        ft_optimizer.zero_grad(); loss.backward(); ft_optimizer.step()
        correct += (out.argmax(1) == labels).sum().item(); total += labels.size(0)
    train_acc = correct/total
    train_acc_hist.append(train_acc)

    clf_model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for imgs, labels in val_loader:
            imgs, labels = imgs.to(device), labels.to(device)
            out = clf_model(imgs)
            correct += (out.argmax(1) == labels).sum().item(); total += labels.size(0)
    val_acc = correct/total
    val_acc_hist.append(val_acc)
    print(f"Epoch {epoch+1}: Train Acc={train_acc:.4f}, Val Acc={val_acc:.4f}")

# Figure 4: Fine-tuning accuracy curves
plt.figure(figsize=(7,5))
plt.plot(train_acc_hist, label="Train Accuracy", marker='o')
plt.plot(val_acc_hist, label="Validation Accuracy", marker='s')
plt.title("Figure 4: Downstream Fine-tuning Accuracy")
plt.xlabel("Epoch"); plt.ylabel("Accuracy"); plt.legend()
plt.tight_layout()
plt.savefig("fig4_finetune_accuracy.png", dpi=400, bbox_inches='tight')
plt.show()

# --------------------------
# 9. FINAL EVALUATION: Accuracy, Precision, Recall, F1, Confusion Matrix
# --------------------------
clf_model.eval()
all_preds, all_labels = [], []
with torch.no_grad():
    for imgs, labels in val_loader:
        imgs = imgs.to(device)
        out = clf_model(imgs)
        preds = out.argmax(1).cpu().numpy()
        all_preds.extend(preds); all_labels.extend(labels.numpy())

acc = accuracy_score(all_labels, all_preds)
precision, recall, f1, _ = precision_recall_fscore_support(all_labels, all_preds, average='weighted')

metrics_table = pd.DataFrame({
    "Metric": ["Accuracy", "Precision", "Recall", "F1-Score"],
    "Value": [acc, precision, recall, f1]
})
print(metrics_table.to_string(index=False))
metrics_table.to_csv("final_metrics_table.csv", index=False)

print("\nClassification Report:\n", classification_report(all_labels, all_preds, target_names=classes[:len(set(all_labels))], zero_division=0))

# Figure 5: Confusion matrix (subset of classes for readability)
cm = confusion_matrix(all_labels, all_preds)
plt.figure(figsize=(12,10))
sns.heatmap(cm, cmap="Blues", cbar=True, xticklabels=False, yticklabels=False)
plt.title("Figure 5: Confusion Matrix on Validation Set")
plt.xlabel("Predicted Label"); plt.ylabel("True Label")
plt.tight_layout()
plt.savefig("fig5_confusion_matrix.png", dpi=400, bbox_inches='tight')
plt.show()

# --------------------------
# 10. FEATURE SPACE VISUALIZATION (t-SNE of learned SSL embeddings)
# --------------------------
model.eval()
embeddings, tsne_labels = [], []
with torch.no_grad():
    for imgs, labels in val_loader:
        imgs = imgs.to(device)
        h, _ = model(imgs)
        embeddings.append(h.cpu().numpy())
        tsne_labels.extend(labels.numpy())
        if len(embeddings) * val_loader.batch_size > 2000:  # cap for speed
            break

embeddings = np.concatenate(embeddings, axis=0)
tsne_labels = np.array(tsne_labels[:len(embeddings)])

tsne = TSNE(n_components=2, random_state=SEED, perplexity=30)
emb_2d = tsne.fit_transform(embeddings)

plt.figure(figsize=(9,8))
scatter = plt.scatter(emb_2d[:,0], emb_2d[:,1], c=tsne_labels, cmap='tab20', s=8, alpha=0.7)
plt.title("Figure 6: t-SNE Visualization of Self-Supervised Feature Embeddings")
plt.xlabel("t-SNE Dim 1"); plt.ylabel("t-SNE Dim 2")
plt.tight_layout()
plt.savefig("fig6_tsne_embeddings.png", dpi=400, bbox_inches='tight')
plt.show()

print("\nAll figures and tables saved to Colab working directory.")
print("Files: fig1_class_distribution.png, fig2_sample_grid.png, fig3_ssl_loss_curve.png,")
print("fig4_finetune_accuracy.png, fig5_confusion_matrix.png, fig6_tsne_embeddings.png,")
print("class_distribution.csv, dataset_summary_table.csv, final_metrics_table.csv")