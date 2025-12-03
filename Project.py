#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# === SETUP & DATA ===
import os
import shutil

def setup_environment():
    print("Installing dependencies...")
    os.system('pip install -q "numpy==1.26.4" "scipy==1.11.4" "rasterio" "albumentations" "segmentation-models-pytorch" "tqdm" "torch_ema"')
    
    if not os.path.exists('mados'):
        print("Cloning repository...")
        os.system('git clone https://github.com/gkakogeorgiou/mados.git')

    if not os.path.exists('./data/MADOS'):
        print("Copying dataset...")
        os.makedirs('./data/MADOS', exist_ok=True)
        src = '/kaggle/input/mados-dataset-new' 
        if os.path.exists(os.path.join(src, 'MADOS')): src = os.path.join(src, 'MADOS')
        os.system(f'cp -r {src}/* ./data/MADOS/')

    if not os.path.exists('./data/MADOS_nearest'):
        print("Stacking bands...")
        os.system('python mados/utils/stack_patches.py --path ./data/MADOS')
        print("Stacking complete.")
    else:
        print("Data ready.")

if __name__ == "__main__":
    setup_environment()


# In[ ]:


# === CELL 2 ===
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
import segmentation_models_pytorch as smp
from segmentation_models_pytorch.losses import LovaszLoss
import rasterio
import numpy as np
import albumentations as A
import os
import random
from glob import glob
from sklearn.model_selection import train_test_split
from torch_ema import ExponentialMovingAverage # <--- NEW: EMA

# --- CONFIGURATION ---
CONFIG = {
    "DEVICE": torch.device("cuda" if torch.cuda.is_available() else "cpu"),
    "BATCH_SIZE": 2,
    "EPOCHS": 40,             # EXTENDED: 40 Epochs
    "NUM_CLASSES": 16,
    "LR": 6e-5,
    "VSCP_PROB": 0.5,
    "PHASE_SWITCH_EPOCH": 15  # Switch to Lovasz later (Epoch 15)
}

print(f"Configuration Loaded. Device: {CONFIG['DEVICE']}")

# --- 1. DATASET CLASS (RARE HUNTER) ---
class MADOSRareHunter(Dataset):
    def __init__(self, file_pair_list, transform=None, crop_size=512, mode='train', bias=0.8):
        self.file_list = file_pair_list
        self.transform = transform
        self.crop_size = crop_size
        self.mode = mode 
        self.bias = bias 
        self.rare_classes = [1, 2, 3, 4, 5, 6, 9, 12, 13, 14, 15]

    def __len__(self): return len(self.file_list)

    def __getitem__(self, idx):
        img_path, mask_path = self.file_list[idx]
        
        with rasterio.open(img_path) as src:
            image = src.read().astype(np.float32)
            image = np.nan_to_num(image)
            img_min, img_max = image.min(), image.max()
            if img_max > img_min:
                image = (image - img_min) / ((img_max - img_min) + 1e-6)
            else:
                image = np.zeros_like(image)

        with rasterio.open(mask_path) as src:
            mask = src.read(1).astype(np.int64)
            
        if self.mode == 'train':
            c, h, w = image.shape
            unique_classes = np.unique(mask)
            present_rare = np.intersect1d(unique_classes, self.rare_classes)
            
            if len(present_rare) > 0 and random.random() < self.bias:
                target_cls = np.random.choice(present_rare)
                indices = np.argwhere(mask == target_cls)
                center = indices[random.randint(0, len(indices)-1)]
                top = max(0, min(h - self.crop_size, center[0] - self.crop_size // 2))
                left = max(0, min(w - self.crop_size, center[1] - self.crop_size // 2))
            else:
                top = random.randint(0, h - self.crop_size) if h > self.crop_size else 0
                left = random.randint(0, w - self.crop_size) if w > self.crop_size else 0
        else:
            c, h, w = image.shape
            top = max(0, (h - self.crop_size) // 2)
            left = max(0, (w - self.crop_size) // 2)

        image = image[:, top:top+self.crop_size, left:left+self.crop_size]
        mask = mask[top:top+self.crop_size, left:left+self.crop_size]
        
        if image.shape[1] < self.crop_size or image.shape[2] < self.crop_size:
            pad_h = max(0, self.crop_size - image.shape[1])
            pad_w = max(0, self.crop_size - image.shape[2])
            image = np.pad(image, ((0,0), (0,pad_h), (0,pad_w)))
            mask = np.pad(mask, ((0,pad_h), (0,pad_w)))

        if self.transform:
            image = np.transpose(image, (1, 2, 0))
            augmented = self.transform(image=image, mask=mask)
            image = augmented['image']
            mask = augmented['mask']
            image = np.transpose(image, (2, 0, 1))
            
        return torch.tensor(image), torch.tensor(mask)

# --- 2. SCENE-BASED SPLITTING (RIGOROUS) ---
def prepare_data():
    print("Organizing Data (Scene Split)...")
    all_pairs = []
    for img_path in glob('./data/MADOS_nearest/**/*_rhorc_*.tif', recursive=True):
        if 'aux' in img_path: continue
        mask_path = img_path.replace('_rhorc_', '_cl_')
        if os.path.exists(mask_path): all_pairs.append((img_path, mask_path))

    scene_ids = [os.path.basename(p[0]).split('_')[1] for p in all_pairs]
    unique_scenes = sorted(list(set(scene_ids)))
    train_scenes, val_scenes = train_test_split(unique_scenes, test_size=0.2, random_state=42)

    train_list = [p for p in all_pairs if os.path.basename(p[0]).split('_')[1] in train_scenes]
    val_list = [p for p in all_pairs if os.path.basename(p[0]).split('_')[1] in val_scenes]
    
    print(f"   Train Scenes: {len(train_scenes)} | Val Scenes: {len(val_scenes)}")
    return train_list, val_list

train_list, val_list = prepare_data()

# --- 3. LOADERS ---
train_transform = A.Compose([
    A.HorizontalFlip(p=0.5), A.VerticalFlip(p=0.5), A.RandomRotate90(p=0.5),
    A.GridDistortion(p=0.3), A.CoarseDropout(max_holes=8, max_height=32, max_width=32, p=0.3)
])
val_transform = A.Compose([A.CenterCrop(512, 512)])

train_ds = MADOSRareHunter(train_list, transform=train_transform, mode='train', bias=0.8)
val_ds = MADOSRareHunter(val_list, transform=val_transform, mode='val')

train_loader = DataLoader(train_ds, batch_size=CONFIG['BATCH_SIZE'], shuffle=True, num_workers=2)
val_loader = DataLoader(val_ds, batch_size=CONFIG['BATCH_SIZE'], shuffle=False, num_workers=2)

# --- 4. MODEL & LOSS ---
print("Building SegFormer...")
model = smp.Segformer(encoder_name="mit_b3", encoder_weights="imagenet", in_channels=11, classes=CONFIG['NUM_CLASSES']).to(CONFIG['DEVICE'])

# EMA SETUP (Exponential Moving Average)
ema = ExponentialMovingAverage(model.parameters(), decay=0.999)

# Loss Configuration
weights = torch.ones(CONFIG['NUM_CLASSES']).float().to(CONFIG['DEVICE'])
for c in [1, 2, 3, 4, 9, 14]: weights[c] = 10.0
for c in [5, 12, 13, 15]: weights[c] = 5.0
weights[6] = 2.0; weights[7] = 1.5; weights[0] = 0.05 # Background enabled

criterion_phase1 = nn.CrossEntropyLoss(weight=weights)
criterion_phase2 = smp.losses.LovaszLoss(mode="multiclass", ignore_index=0)

optimizer = AdamW(model.parameters(), lr=CONFIG['LR'])
scheduler = CosineAnnealingLR(optimizer, T_max=CONFIG['EPOCHS'])

print("Scientific Setup Complete: Scene Split + EMA + True VSCP.")


# In[ ]:


# === TRAINING LOOP  ===
import sys
import random
from tqdm import tqdm
import torch.nn.utils
from torch.cuda.amp import autocast, GradScaler 

def apply_vscp_batch(images, masks):
    batch_size = images.shape[0]
    if batch_size < 2: return images, masks
    
    half = batch_size // 2
    imgs_A = images[:half].clone()
    imgs_B = images[half:].clone()
    masks_A = masks[:half].clone()
    masks_B = masks[half:].clone()
    
    pixels_to_copy = (masks_B > 0)
    mask_expanded = pixels_to_copy.unsqueeze(1).expand_as(imgs_A)
    
    imgs_A[mask_expanded] = imgs_B[mask_expanded]
    masks_A[pixels_to_copy] = masks_B[pixels_to_copy]
    
    images[:half] = imgs_A
    masks[:half] = masks_A
    
    return images, masks

MAX_GRAD_NORM = 1.0
VSCP_PROB = 0.5 
scaler = torch.amp.GradScaler('cuda')

best_loss_p1 = float('inf')
best_loss_p2 = float('inf')

print(f"Starting Training ({CONFIG['EPOCHS']} Epochs)...")

for epoch in range(CONFIG['EPOCHS']):
    model.train()
    train_loss = 0
    loop = tqdm(train_loader, desc=f"Epoch {epoch+1}/{CONFIG['EPOCHS']}")
    
    if epoch < CONFIG['PHASE_SWITCH_EPOCH']:
        criterion = criterion_phase1
        phase_name = "Hunter (CE)"
    else:
        criterion = criterion_phase2
        phase_name = "Sculptor (Lovasz)"
    loop.set_description(f"Ep {epoch+1} [{phase_name}]")
    
    for batch_idx, (images, masks) in enumerate(loop):
        images, masks = images.to(CONFIG['DEVICE']), masks.to(CONFIG['DEVICE']).long()
        masks[masks >= CONFIG['NUM_CLASSES']] = 0
        
        if random.random() < CONFIG['VSCP_PROB']:
            images, masks = apply_vscp_batch(images, masks)
        
        with autocast():
            outputs = model(images)
            loss = criterion(outputs, masks)
        
        if torch.isnan(loss):
            print(f"\nNaN detected!"); sys.exit()
            
        optimizer.zero_grad()
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), MAX_GRAD_NORM)
        scaler.step(optimizer)
        scaler.update()
        
        ema.update()
        
        train_loss += loss.item()
        loop.set_postfix(loss=loss.item())
    
    val_loss = 0
    with torch.no_grad():
        with ema.average_parameters():
            for images, masks in val_loader:
                images, masks = images.to(CONFIG['DEVICE']), masks.to(CONFIG['DEVICE']).long()
                masks[masks >= CONFIG['NUM_CLASSES']] = 0
                with autocast():
                    outputs = model(images)
                    val_loss += criterion(outputs, masks).item()
    
    avg_val = val_loss / len(val_loader)
    scheduler.step()
    
    print(f"    {phase_name} Val Loss (EMA): {avg_val:.4f}")
    
    if epoch < CONFIG['PHASE_SWITCH_EPOCH']:
        if avg_val < best_loss_p1:
            best_loss_p1 = avg_val
            with ema.average_parameters():
                torch.save(model.state_dict(), "segformer_phase1_hunter.pth")
            print("    Phase 1 EMA Model Updated!")
    else:
        if avg_val < best_loss_p2:
            best_loss_p2 = avg_val
            with ema.average_parameters():
                torch.save(model.state_dict(), "segformer_phase2_sculptor.pth")
            print("    Phase 2 EMA Model Updated!")

print("Training Finished!")


# # === CELL 3b: RESUME TRAINING (BUG FIXED) ===
# import torch
# import sys
# import random
# from tqdm import tqdm

# print("Resuming from Phase 2 Checkpoint...")
# if os.path.exists("segformer_phase2_sculptor.pth"):
#     model.load_state_dict(torch.load("segformer_phase2_sculptor.pth"))
#     print("Loaded 'segformer_phase2_sculptor.pth'. Resuming refinement...")
# else:
#     print("Checkpoint not found. Starting Phase 2 from scratch (using current weights).")

# START_EPOCH = 23 
# TOTAL_EPOCHS = 40
# criterion = criterion_phase2
# phase_name = "Sculptor (Resume)"

# for epoch in range(START_EPOCH, TOTAL_EPOCHS):
#     model.train()
#     train_loss = 0
#     loop = tqdm(train_loader, desc=f"Epoch {epoch+1}/{TOTAL_EPOCHS}")
    
#     for batch_idx, (images, masks) in enumerate(loop):
#         images, masks = images.to(CONFIG['DEVICE']), masks.to(CONFIG['DEVICE']).long()
#         masks[masks >= CONFIG['NUM_CLASSES']] = 0
        
#         if random.random() < CONFIG['VSCP_PROB']:
#             images, masks = apply_vscp_batch(images, masks)
        
#         with autocast():
#             outputs = model(images)
#             loss = criterion(outputs, masks)
        
#         if torch.isnan(loss).any():
#             print(f"\nNaN detected!"); sys.exit()

#         optimizer.zero_grad()
#         scaler.scale(loss).backward()
#         scaler.unscale_(optimizer)
#         torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
#         scaler.step(optimizer)
#         scaler.update()
        
#         ema.update()
        
#         train_loss += loss.item()
#         loop.set_postfix(loss=loss.item())
    
#     val_loss = 0
#     with torch.no_grad():
#         with ema.average_parameters():
#             for images, masks in val_loader:
#                 images, masks = images.to(CONFIG['DEVICE']), masks.to(CONFIG['DEVICE']).long()
#                 masks[masks >= CONFIG['NUM_CLASSES']] = 0
#                 with autocast():
#                     outputs = model(images)
#                     val_loss += criterion(outputs, masks).item()
    
#     avg_val = val_loss / len(val_loader)
#     scheduler.step()
#     print(f"   📉 {phase_name} Val Loss (EMA): {avg_val:.4f}")
    
#     if avg_val < best_loss_p2:
#         best_loss_p2 = avg_val
#         with ema.average_parameters():
#             torch.save(model.state_dict(), "segformer_phase2_sculptor.pth")
#         print("   Phase 2 EMA Model Updated!")

# print("Resume Finished!")


# In[ ]:


# === EVALUATION ===
import torch
import numpy as np
import pandas as pd
from tqdm import tqdm
import seaborn as sns
import matplotlib.pyplot as plt
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

CLASS_NAMES = {
    0: "Background", 1: "Marine Debris (Plastic)", 2: "Dense Sargassum",
    3: "Sparse Floating Algae", 4: "Natural Organic Material", 5: "Ship",
    6: "Oil Spill", 7: "Marine Water", 8: "Sediment-Laden Water",
    9: "Foam", 10: "Turbid Water", 11: "Shallow Water",
    12: "Waves & Wakes", 13: "Oil Platform", 14: "Jellyfish", 15: "Sea Snot"
}

class MADOSUnseenDataset(Dataset):
    def __init__(self, file_list): self.file_list = file_list
    def __len__(self): return len(self.file_list)
    def __getitem__(self, idx):
        img_path, mask_path = self.file_list[idx]
        with rasterio.open(img_path) as src:
            image = src.read().astype(np.float32)
            image = np.nan_to_num(image)
            if image.max() > image.min():
                image = (image - image.min()) / ((image.max() - image.min()) + 1e-6)
        with rasterio.open(mask_path) as src:
            mask = src.read(1).astype(np.int64)
        
        c, h, w = image.shape
        top, left = max(0, (h - 512) // 2), max(0, (w - 512) // 2)
        image = image[:, top:top+512, left:left+512]
        mask = mask[top:top+512, left:left+512]
        
        if image.shape[1] < 512 or image.shape[2] < 512:
            pad_h, pad_w = max(0, 512 - image.shape[1]), max(0, 512 - image.shape[2])
            image = np.pad(image, ((0,0), (0,pad_h), (0,pad_w)))
            mask = np.pad(mask, ((0,pad_h), (0,pad_w)))
            
        return torch.tensor(image), torch.tensor(mask)

def predict_with_tta(model, image):
    model.eval()
    logits_list = []
    rotations = [0, 1, 2, 3]
    with torch.no_grad():
        for k in rotations:
            img_rot = torch.rot90(image, k=k, dims=[2, 3])
            out_rot = model(img_rot)
            logits_list.append(torch.rot90(out_rot, k=-k, dims=[2, 3]))
            
            img_flip = torch.flip(img_rot, dims=[3])
            out_flip = model(img_flip)
            out_unflip = torch.flip(out_flip, dims=[3])
            logits_list.append(torch.rot90(out_unflip, k=-k, dims=[2, 3]))
            
    return torch.mean(torch.stack(logits_list), dim=0)

def evaluate_rigorous(model, file_list, device, num_classes):
    ds = MADOSUnseenDataset(file_list)
    loader = DataLoader(ds, batch_size=4, shuffle=False, num_workers=2)
    
    cm = torch.zeros((num_classes, num_classes), dtype=torch.int64, device=device)
    print(f"📊 Evaluating...")
    
    with torch.no_grad():
        for images, masks in tqdm(loader):
            images = images.to(device)
            masks = masks.to(device).long()
            
            outputs = predict_with_tta(model, images)
            preds = torch.argmax(outputs, dim=1)
            
            valid = (masks != 0)
            if valid.sum() == 0: continue
            
            y_true, y_pred = masks[valid].flatten(), preds[valid].flatten()
            bincount = torch.bincount(num_classes * y_true + y_pred, minlength=num_classes**2)
            cm += bincount.reshape(num_classes, num_classes)
            
    return cm

if 'val_list' not in locals():
    print("Error: 'val_list' missing. Run Cell 2 first.")
else:
    conf_mat = evaluate_rigorous(model, val_list, CONFIG["DEVICE"], CONFIG["NUM_CLASSES"])
    cm = conf_mat.cpu().numpy()
    
    TP = np.diag(cm)
    FP = cm.sum(axis=0) - TP
    FN = cm.sum(axis=1) - TP
    eps = 1e-6
    
    IoU = TP / (TP + FP + FN + eps)
    F1 = 2 * TP / (2 * TP + FP + FN + eps)
    
    print("\n" + "="*60)
    print("FINAL RESULTS")
    print("="*60)
    print(f"mIoU (1-15): {np.nanmean(IoU[1:]):.4f}")
    print(f"Mean F1 (1-15):  {np.nanmean(F1[1:]):.4f}")
    print("-" * 60)
    
    stats = []
    for i in range(1, CONFIG["NUM_CLASSES"]):
        stats.append({
            "Class": CLASS_NAMES[i], 
            "IoU": IoU[i], 
            "F1": F1[i], 
            "Pixels": cm[i, :].sum()
        })
        
    print(pd.DataFrame(stats).sort_values(by="IoU", ascending=False).to_string(index=False, formatters={"IoU": "{:.4f}".format, "F1": "{:.4f}".format}))

      plt.figure(figsize=(14, 10))
    cm_norm = cm.astype('float') / (cm.sum(axis=1)[:, np.newaxis] + 1e-6)
    
    sns.heatmap(cm_norm[1:, 1:], annot=True, fmt=".2f", cmap="Blues",
                xticklabels=[CLASS_NAMES[i] for i in range(1, 16)],
                yticklabels=[CLASS_NAMES[i] for i in range(1, 16)])
    
    plt.title("Normalized Confusion Matrix")
    plt.ylabel("True Label")
    plt.xlabel("Predicted Label")
    plt.xticks(rotation=45, ha='right')
    plt.show()


# In[ ]:


# === VISUALIZATION ===
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.colors as mcolors
import torch.nn.functional as F
from torch.utils.data import Dataset
import rasterio
import numpy as np
from glob import glob
import os

CONFIDENCE_THRESHOLD = 0.85
DEBRIS_CLASSES = [1, 2, 3, 4, 5, 6, 9, 13, 14, 15]

class MADOSFullSceneDataset(Dataset):
    def __init__(self, root_dir):
        self.samples = []
        for img_path in glob(os.path.join(root_dir, '**', '*_rhorc_*.tif'), recursive=True):
            if 'aux' in img_path: continue
            mask_path = img_path.replace('_rhorc_', '_cl_')
            if os.path.exists(mask_path): self.samples.append((img_path, mask_path))
    def __len__(self): return len(self.samples)
    def __getitem__(self, idx): return self.samples[idx]

def predict_with_tta(model, image):
    model.eval()
    logits_list = []
    with torch.no_grad():
        for k in [0, 1, 2, 3]:
            rot = torch.rot90(image, k=k, dims=[2, 3])
            out = model(rot)
            logits_list.append(torch.rot90(out, k=-k, dims=[2, 3]))
            flip = torch.flip(rot, dims=[3])
            out_flip = model(flip)
            logits_list.append(torch.rot90(torch.flip(out_flip, dims=[3]), k=-k, dims=[2, 3]))
    return torch.mean(torch.stack(logits_list), dim=0)

def predict_sliding_window(model, image_tensor, max_tile_size=512, overlap=1/3):
    model.eval()
    _, c, h, w = image_tensor.shape
    
    tile_size = 256 if (h < max_tile_size or w < max_tile_size) else max_tile_size
    
    target_h, target_w = ((h+31)//32)*32, ((w+31)//32)*32
    pad_h, pad_w = target_h - h, target_w - w
    if pad_h > 0 or pad_w > 0:
        image_tensor = F.pad(image_tensor, (0, pad_w, 0, pad_h), mode='constant', value=0)
        
    padded_h, padded_w = image_tensor.shape[2:]
    stride = int(tile_size * (1 - overlap))
    
    heatmap = torch.zeros((1, CONFIG["NUM_CLASSES"], padded_h, padded_w), device=CONFIG["DEVICE"])
    countmap = torch.zeros((1, 1, padded_h, padded_w), device=CONFIG["DEVICE"])
    
    with torch.no_grad():
        for y in range(0, padded_h, stride):
            for x in range(0, padded_w, stride):
                y1, x1 = min(y, padded_h - tile_size), min(x, padded_w - tile_size)
                y2, x2 = y1 + tile_size, x1 + tile_size
                
                tile = image_tensor[:, :, y1:y2, x1:x2]
                output = predict_with_tta(model, tile)
                
                heatmap[:, :, y1:y2, x1:x2] += output
                countmap[:, :, y1:y2, x1:x2] += 1.0
                
    heatmap /= countmap
    
    probs = F.softmax(heatmap, dim=1)
    max_probs, preds = torch.max(probs, dim=1)
    preds[max_probs < CONFIDENCE_THRESHOLD] = 0
    
    return preds.squeeze().cpu().numpy()[:h, :w]

def visualize_full_scene(dataset, idx):
    img_path, mask_path = dataset[idx]
    filename = os.path.basename(img_path)
    
    with rasterio.open(img_path) as src:
        image = src.read().astype(np.float32)
        image = np.nan_to_num(image)
        if image.max() > image.min(): 
            image = (image - image.min()) / ((image.max() - image.min()) + 1e-6)
            
    with rasterio.open(mask_path) as src: 
        mask = src.read(1).astype(np.int64)
    
    print(f"\nProcessing {filename}")
    
     img_tensor = torch.tensor(image).unsqueeze(0).to(CONFIG["DEVICE"])
    pred_mask = predict_sliding_window(model, img_tensor)
    
    debris_mask = pred_mask.copy()
    is_debris = np.isin(debris_mask, DEBRIS_CLASSES)
    debris_mask[~is_debris] = 0
    
    fig, ax = plt.subplots(1, 4, figsize=(32, 8))
    
    rgb = image[0:3].transpose(1, 2, 0)
    rgb = (rgb - rgb.min()) / (rgb.max() - rgb.min())
    
    cmap = plt.get_cmap("tab20")
    cmap_list = [cmap(i) for i in range(cmap.N)]; cmap_list[0] = (0, 0, 0, 1)
    custom_cmap = mcolors.ListedColormap(cmap_list)
    
    ax[0].imshow(rgb); ax[0].set_title("Input RGB"); ax[0].axis('off')
    ax[1].imshow(mask, cmap=custom_cmap, vmin=0, vmax=20, interpolation='nearest')
    ax[1].set_title("Ground Truth"); ax[1].axis('off')
    ax[2].imshow(pred_mask, cmap=custom_cmap, vmin=0, vmax=20, interpolation='nearest')
    ax[2].set_title(f"Prediction (Confidence > {CONFIDENCE_THRESHOLD})"); ax[2].axis('off')
    ax[3].imshow(debris_mask, cmap=custom_cmap, vmin=0, vmax=20, interpolation='nearest')
    ax[3].set_title("Targeted Debris Extraction"); ax[3].axis('off')
    
    unique = np.unique(np.concatenate((mask, pred_mask)))
    patches = []
    for c in unique:
        if c == 0: continue
        color = cmap_list[c] if c < 20 else (1,1,1,1)
        patches.append(mpatches.Patch(color=color, label=f"{c}: {CLASS_NAMES.get(c,'?')}"))
    
    if patches: 
        ax[3].legend(handles=patches, bbox_to_anchor=(1.05, 1), loc='upper left')
    
    plt.tight_layout()
    plt.show()

print("Running Visualization...")
full_scene_ds = MADOSFullSceneDataset('./data/MADOS_nearest')
for i in range(3):
    idx = np.random.randint(0, len(full_scene_ds))
    visualize_full_scene(full_scene_ds, idx)


# In[ ]:


# === CELL 6: MINIMAL VISUALIZATION (FIXED VARIABLE NAMES) ===
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.colors as mcolors
import torch.nn.functional as F
from torch.utils.data import Dataset
import rasterio
import numpy as np
from glob import glob
import os
from skimage import exposure

# --- CONFIGURATION ---
CONFIDENCE_THRESHOLD = 0.85
DEBRIS_CLASSES = [1, 2, 3, 4, 5, 6, 9, 13, 14, 15]

CLASS_NAMES = {
    0: "Background", 1: "Marine Debris (Plastic)", 2: "Dense Sargassum",
    3: "Sparse Floating Algae", 4: "Natural Organic Material", 5: "Ship",
    6: "Oil Spill", 7: "Marine Water", 8: "Sediment-Laden Water",
    9: "Foam", 10: "Turbid Water", 11: "Shallow Water",
    12: "Waves & Wakes", 13: "Oil Platform", 14: "Jellyfish", 15: "Sea Snot"
}

# --- DATASET & INFERENCE UTILS ---
class MADOSFullSceneDataset(Dataset):
    def __init__(self, root_dir):
        self.samples = []
        for img_path in glob(os.path.join(root_dir, '**', '*_rhorc_*.tif'), recursive=True):
            if 'aux' in img_path: continue
            mask_path = img_path.replace('_rhorc_', '_cl_')
            if os.path.exists(mask_path): self.samples.append((img_path, mask_path))
    def __len__(self): return len(self.samples)
    def __getitem__(self, idx): return self.samples[idx]

def predict_with_tta(model, image):
    model.eval()
    logits_list = []
    with torch.no_grad():
        for k in [0, 1, 2, 3]:
            rot = torch.rot90(image, k=k, dims=[2, 3])
            out = model(rot)
            logits_list.append(torch.rot90(out, k=-k, dims=[2, 3]))
            flip = torch.flip(rot, dims=[3])
            out_flip = model(flip)
            logits_list.append(torch.rot90(torch.flip(out_flip, dims=[3]), k=-k, dims=[2, 3]))
    return torch.mean(torch.stack(logits_list), dim=0)

def predict_sliding_window(model, image_tensor, max_tile_size=512, overlap=1/3):
    model.eval()
    _, c, h, w = image_tensor.shape
    tile_size = 256 if (h < max_tile_size or w < max_tile_size) else max_tile_size
    target_h, target_w = ((h+31)//32)*32, ((w+31)//32)*32
    pad_h, pad_w = target_h - h, target_w - w
    if pad_h > 0 or pad_w > 0:
        image_tensor = F.pad(image_tensor, (0, pad_w, 0, pad_h), mode='constant', value=0)
    padded_h, padded_w = image_tensor.shape[2:]
    stride = int(tile_size * (1 - overlap))
    
    # --- FIX: Use CONFIG Dictionary ---
    heatmap = torch.zeros((1, CONFIG['NUM_CLASSES'], padded_h, padded_w), device=CONFIG['DEVICE'])
    countmap = torch.zeros((1, 1, padded_h, padded_w), device=CONFIG['DEVICE'])
    
    with torch.no_grad():
        for y in range(0, padded_h, stride):
            for x in range(0, padded_w, stride):
                y1, x1 = min(y, padded_h - tile_size), min(x, padded_w - tile_size)
                y2, x2 = y1 + tile_size, x1 + tile_size
                tile = image_tensor[:, :, y1:y2, x1:x2]
                output = predict_with_tta(model, tile)
                heatmap[:, :, y1:y2, x1:x2] += output
                countmap[:, :, y1:y2, x1:x2] += 1.0
    heatmap /= countmap
    probs = F.softmax(heatmap, dim=1)
    max_probs, preds = torch.max(probs, dim=1)
    preds[max_probs < CONFIDENCE_THRESHOLD] = 0
    return preds.squeeze().cpu().numpy()[:h, :w]

# --- VISUALIZATION ---
def visualize_minimal_scene(dataset, idx):
    img_path, mask_path = dataset[idx]
    filename = os.path.basename(img_path)
    
    with rasterio.open(img_path) as src:
        raw_image = src.read().astype(np.float32)
    
    print(f"\n📸 Processing {filename}...")
    
    # Model Input Norm
    model_input = raw_image.copy()
    model_input = np.nan_to_num(model_input)
    if model_input.max() > model_input.min():
        model_input = (model_input - model_input.min()) / ((model_input.max() - model_input.min()) + 1e-6)
            
    # Inference
    img_tensor = torch.tensor(model_input).unsqueeze(0).to(CONFIG['DEVICE']) # FIX: Use CONFIG
    pred_mask = predict_sliding_window(model, img_tensor)
    
    # Debris Mask
    debris_mask = pred_mask.copy()
    is_debris = np.isin(debris_mask, DEBRIS_CLASSES)
    debris_mask[~is_debris] = 0
    
    if np.sum(is_debris) == 0:
        print("   (Skipping visualization - No debris detected)")
        
        
    # RGB Processing
    rgb = raw_image[[2, 1, 0], :, :].transpose(1, 2, 0)
    p2, p98 = np.percentile(rgb, (2, 98))
    rgb = np.clip((rgb - p2) / (p98 - p2), 0, 1)
    rgb = exposure.adjust_gamma(rgb, 0.8)

    # Plotting
    fig, ax = plt.subplots(1, 2, figsize=(16, 8))
    
    cmap = plt.get_cmap("tab20")
    cmap_list = [cmap(i) for i in range(cmap.N)]; cmap_list[0] = (0, 0, 0, 1)
    custom_cmap = mcolors.ListedColormap(cmap_list)
    
    ax[0].imshow(rgb); ax[0].set_title("Satellite Input (Enhanced RGB)"); ax[0].axis('off')
    ax[1].imshow(debris_mask, cmap=custom_cmap, vmin=0, vmax=20, interpolation='nearest')
    ax[1].set_title(f"Detected Debris (Conf > {CONFIDENCE_THRESHOLD})"); ax[1].axis('off')
    
    unique = np.unique(debris_mask)
    patches = [mpatches.Patch(color=cmap_list[c], label=f"{c}: {CLASS_NAMES.get(c,'?')}") for c in unique if c!=0]
    if patches: ax[1].legend(handles=patches, bbox_to_anchor=(1.05, 1), loc='upper left')
    
    plt.tight_layout(); plt.show()

# Execute
print("👀 Searching for scenes with debris...")
full_scene_ds = MADOSFullSceneDataset('./data/MADOS_nearest')
count = 0
for _ in range(100): 
    idx = np.random.randint(0, len(full_scene_ds))
    visualize_minimal_scene(full_scene_ds, idx)
    # Check if a plot was actually created (matplotlib keeps track of open figures)
    if plt.get_fignums(): 
        count += 1
        if count >= 3: break

