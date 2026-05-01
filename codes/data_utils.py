import random
import warnings

import numpy as np
import pandas as pd
from PIL import Image, ImageFile

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models

from sklearn.metrics import accuracy_score, f1_score
from tqdm.auto import tqdm

from config import (
    SEED,
    IMG_SIZE,
    BATCH_SIZE,
    IMAGE_EXTS,
    DATA_DIR,
    DISPLAY_NAME_MAP,
)

warnings.filterwarnings("ignore")
ImageFile.LOAD_TRUNCATED_IMAGES = True


def set_seed(seed=SEED):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def get_device():
    return "cuda" if torch.cuda.is_available() else "cpu"


def build_manifest(data_dir=DATA_DIR):
    image_paths = [
        p for p in data_dir.rglob("*")
        if p.is_file() and p.suffix.lower() in IMAGE_EXTS
    ]

    rows = []

    for path in image_paths:
        rel_parent = path.parent.relative_to(data_dir)

        if str(rel_parent) == ".":
            continue

        label = str(rel_parent).replace("\\", "/")
        rows.append({"path": str(path), "label": label})

    df = pd.DataFrame(rows)

    if df.empty:
        raise ValueError(
            "No labeled images found. Expected folders like handbags_data/class_name/image.jpg"
        )

    labels = sorted(df["label"].unique())
    label_to_idx = {label: i for i, label in enumerate(labels)}
    idx_to_label = {i: label for label, i in label_to_idx.items()}

    df["label_id"] = df["label"].map(label_to_idx)
    df["display_label"] = df["label"].map(DISPLAY_NAME_MAP).fillna(df["label"])

    return df, labels, label_to_idx, idx_to_label


def stratified_split(dataframe, train_frac=0.70, val_frac=0.15, test_frac=0.15, seed=SEED):
    assert abs(train_frac + val_frac + test_frac - 1.0) < 1e-6

    split_frames = []

    for label, group in dataframe.groupby("label"):
        group = group.sample(frac=1, random_state=seed).reset_index(drop=True)
        n = len(group)

        if n < 5:
            group["split"] = "train"
        else:
            n_train = max(1, int(round(n * train_frac)))
            n_val = max(1, int(round(n * val_frac)))
            n_test = n - n_train - n_val

            if n_test < 1:
                n_test = 1
                n_train = n - n_val - n_test

            group.loc[:n_train - 1, "split"] = "train"
            group.loc[n_train:n_train + n_val - 1, "split"] = "val"
            group.loc[n_train + n_val:, "split"] = "test"

        split_frames.append(group)

    df_split = pd.concat(split_frames, ignore_index=True)
    df_split["display_label"] = df_split["label"].map(DISPLAY_NAME_MAP).fillna(df_split["label"])

    train_df = df_split[df_split["split"] == "train"].reset_index(drop=True)
    val_df = df_split[df_split["split"] == "val"].reset_index(drop=True)
    test_df = df_split[df_split["split"] == "test"].reset_index(drop=True)

    return df_split, train_df, val_df, test_df


def get_transforms():
    train_transform = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(10),
        transforms.ColorJitter(brightness=0.15, contrast=0.15, saturation=0.15),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
        ),
    ])

    eval_transform = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
        ),
    ])

    return train_transform, eval_transform


class HandbagDataset(Dataset):
    def __init__(self, dataframe, transform=None):
        self.dataframe = dataframe.reset_index(drop=True)
        self.transform = transform

    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, idx):
        row = self.dataframe.iloc[idx]
        image = Image.open(row["path"]).convert("RGB")

        if self.transform:
            image = self.transform(image)

        label = int(row["label_id"])
        return image, label


class ImageOnlyDataset(Dataset):
    def __init__(self, paths, transform=None):
        self.paths = list(paths)
        self.transform = transform

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        path = self.paths[idx]
        image = Image.open(path).convert("RGB")

        if self.transform:
            image = self.transform(image)

        return image, path


def build_dataloaders(train_df, val_df, test_df):
    train_transform, eval_transform = get_transforms()

    train_ds = HandbagDataset(train_df, transform=train_transform)
    val_ds = HandbagDataset(val_df, transform=eval_transform)
    test_ds = HandbagDataset(test_df, transform=eval_transform)

    train_loader = DataLoader(
        train_ds,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=0,
        pin_memory=torch.cuda.is_available(),
    )

    val_loader = DataLoader(
        val_ds,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=0,
        pin_memory=torch.cuda.is_available(),
    )

    test_loader = DataLoader(
        test_ds,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=0,
        pin_memory=torch.cuda.is_available(),
    )

    return train_loader, val_loader, test_loader


def build_model(num_classes, freeze_backbone=True):
    try:
        weights = models.ResNet18_Weights.DEFAULT
        model = models.resnet18(weights=weights)
        print("Loaded pretrained ResNet18.")
    except Exception as e:
        print("Could not load pretrained weights. Using random initialization.")
        print("Reason:", e)
        model = models.resnet18(weights=None)

    if freeze_backbone:
        for param in model.parameters():
            param.requires_grad = False

    in_features = model.fc.in_features
    model.fc = nn.Sequential(
        nn.Dropout(0.25),
        nn.Linear(in_features, num_classes),
    )

    return model


def make_class_weighted_loss(train_df, num_classes, device):
    train_counts = train_df["label_id"].value_counts().reindex(
        range(num_classes),
        fill_value=1,
    ).sort_index()

    class_weights = 1.0 / torch.tensor(train_counts.values, dtype=torch.float32)
    class_weights = class_weights / class_weights.sum() * num_classes
    class_weights = class_weights.to(device)

    return nn.CrossEntropyLoss(weight=class_weights)


def run_epoch(model, loader, criterion, device, optimizer=None):
    is_train = optimizer is not None
    model.train() if is_train else model.eval()

    total_loss = 0.0
    all_true = []
    all_pred = []

    for images, labels_batch in tqdm(loader, leave=False):
        images = images.to(device)
        labels_batch = labels_batch.to(device)

        if is_train:
            optimizer.zero_grad()

        with torch.set_grad_enabled(is_train):
            logits = model(images)
            loss = criterion(logits, labels_batch)

            if is_train:
                loss.backward()
                optimizer.step()

        total_loss += loss.item() * images.size(0)

        preds = logits.argmax(dim=1).detach().cpu().numpy()
        truths = labels_batch.detach().cpu().numpy()

        all_pred.extend(preds)
        all_true.extend(truths)

    avg_loss = total_loss / len(loader.dataset)
    acc = accuracy_score(all_true, all_pred)
    macro_f1 = f1_score(all_true, all_pred, average="macro", zero_division=0)

    return avg_loss, acc, macro_f1


def collect_predictions(model, loader, dataframe, idx_to_label, device):
    model.eval()

    all_true = []
    all_pred = []
    all_probs = []

    with torch.no_grad():
        for images, labels_batch in tqdm(loader):
            images = images.to(device)
            logits = model(images)
            probs = torch.softmax(logits, dim=1)

            preds = probs.argmax(dim=1).cpu().numpy()
            probs = probs.cpu().numpy()
            truths = labels_batch.numpy()

            all_true.extend(truths)
            all_pred.extend(preds)
            all_probs.extend(probs)

    result = dataframe.copy()
    result["true_id"] = all_true
    result["pred_id"] = all_pred

    result["Actual Category"] = (
        result["true_id"]
        .map(idx_to_label)
        .map(DISPLAY_NAME_MAP)
        .fillna(result["true_id"].map(idx_to_label))
    )

    result["Predicted Category"] = (
        result["pred_id"]
        .map(idx_to_label)
        .map(DISPLAY_NAME_MAP)
        .fillna(result["pred_id"].map(idx_to_label))
    )

    result["Confidence"] = [float(np.max(p)) for p in all_probs]

    return result, np.array(all_true), np.array(all_pred), np.array(all_probs)


def build_feature_extractor(trained_model, device):
    feature_extractor = nn.Sequential(*list(trained_model.children())[:-1])
    feature_extractor.to(device)
    feature_extractor.eval()
    return feature_extractor


def extract_features(paths, feature_extractor, eval_transform, device, batch_size=32):
    dataset = ImageOnlyDataset(paths, transform=eval_transform)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0)

    all_features = []
    all_paths = []

    with torch.no_grad():
        for images, batch_paths in tqdm(loader):
            images = images.to(device)
            feats = feature_extractor(images)
            feats = feats.view(feats.size(0), -1)
            feats = torch.nn.functional.normalize(feats, p=2, dim=1)

            all_features.append(feats.cpu().numpy())
            all_paths.extend(batch_paths)

    features = np.vstack(all_features)
    return features, list(all_paths)
