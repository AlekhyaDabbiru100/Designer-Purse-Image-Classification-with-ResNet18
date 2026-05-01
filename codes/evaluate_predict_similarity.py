import json
from pathlib import Path
import math

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image

import torch
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay,
)

from config import (
    SEED,
    DATA_DIR,
    OUTPUT_DIR,
    FINE_TUNE_CHECKPOINT_PATH,
    BASE_CHECKPOINT_PATH,
    DISPLAY_NAME_MAP,
)

from data_utils import (
    set_seed,
    get_device,
    build_manifest,
    stratified_split,
    build_dataloaders,
    build_model,
    collect_predictions,
    get_transforms,
    build_feature_extractor,
    extract_features,
)


def save_confusion_matrix(cm, target_names, filename, normalized=False):
    fig, ax = plt.subplots(figsize=(10, 10))

    if normalized:
        cm_display = cm.astype("float") / cm.sum(axis=1, keepdims=True)
        values_format = ".2f"
        title = "Normalized Confusion Matrix for Purse Classification"
    else:
        cm_display = cm
        values_format = "d"
        title = "Confusion Matrix for Purse Classification"

    disp = ConfusionMatrixDisplay(
        confusion_matrix=cm_display,
        display_labels=target_names,
    )

    disp.plot(ax=ax, xticks_rotation=45, colorbar=normalized, values_format=values_format)
    plt.title(title)
    plt.xlabel("Predicted Category")
    plt.ylabel("Actual Category")
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / filename, dpi=150)
    plt.close()


def predict_image(image_path, model, eval_transform, idx_to_label, device, top_k=5):
    image_path = Path(image_path)
    image = Image.open(image_path).convert("RGB")
    x = eval_transform(image).unsqueeze(0).to(device)

    model.eval()
    with torch.no_grad():
        logits = model(x)
        probs = torch.softmax(logits, dim=1).squeeze(0).cpu().numpy()

    top_indices = probs.argsort()[::-1][:top_k]

    results = pd.DataFrame({
        "Category": [
            DISPLAY_NAME_MAP.get(idx_to_label[i], idx_to_label[i])
            for i in top_indices
        ],
        "Probability": [float(probs[i]) for i in top_indices],
    })

    return results


def find_similar_images(query_path, df, model, eval_transform, device, top_k=8):
    feature_extractor = build_feature_extractor(model, device)

    bank_paths = df["path"].tolist()
    bank_features, bank_paths = extract_features(
        bank_paths,
        feature_extractor,
        eval_transform,
        device,
        batch_size=32,
    )

    path_to_label = dict(zip(df["path"], df["label"]))

    query_path = str(query_path)
    query_features, _ = extract_features(
        [query_path],
        feature_extractor,
        eval_transform,
        device,
        batch_size=1,
    )

    similarities = bank_features @ query_features[0]
    ranked_indices = np.argsort(similarities)[::-1]

    ranked_indices = [
        i for i in ranked_indices
        if Path(bank_paths[i]).resolve() != Path(query_path).resolve()
    ]

    top_indices = ranked_indices[:top_k]

    results = pd.DataFrame({
        "path": [bank_paths[i] for i in top_indices],
        "Category": [
            DISPLAY_NAME_MAP.get(path_to_label.get(bank_paths[i], ""), path_to_label.get(bank_paths[i], "Unknown"))
            for i in top_indices
        ],
        "Similarity Score": [float(similarities[i]) for i in top_indices],
    })

    return results


def main():
    set_seed(SEED)
    device = get_device()

    df, labels, label_to_idx, idx_to_label = build_manifest(DATA_DIR)
    num_classes = len(labels)
    _, train_df, val_df, test_df = stratified_split(df, seed=SEED)
    _, _, test_loader = build_dataloaders(train_df, val_df, test_df)
    _, eval_transform = get_transforms()

    target_names = [
        DISPLAY_NAME_MAP.get(idx_to_label[i], idx_to_label[i])
        for i in range(num_classes)
    ]

    checkpoint_path = FINE_TUNE_CHECKPOINT_PATH if FINE_TUNE_CHECKPOINT_PATH.exists() else BASE_CHECKPOINT_PATH
    checkpoint = torch.load(checkpoint_path, map_location=device)

    model = build_model(
        num_classes,
        freeze_backbone=checkpoint.get("freeze_backbone", False),
    ).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    print("Loaded checkpoint from:", checkpoint_path)

    test_preds_df, y_true, y_pred, y_probs = collect_predictions(
        model,
        test_loader,
        test_df,
        idx_to_label,
        device,
    )

    test_acc = accuracy_score(y_true, y_pred)
    test_macro_f1 = f1_score(y_true, y_pred, average="macro", zero_division=0)

    print("Test Accuracy:", round(test_acc, 4))
    print("Test Macro F1:", round(test_macro_f1, 4))

    report_dict = classification_report(
        y_true,
        y_pred,
        target_names=target_names,
        output_dict=True,
        zero_division=0,
    )

    report_df = pd.DataFrame(report_dict).transpose()
    print("\nClassification Report:")
    print(report_df)

    test_preds_df.to_csv(OUTPUT_DIR / "test_predictions.csv", index=False)
    report_df.to_csv(OUTPUT_DIR / "fine_tuned_classification_report.csv")

    cm = confusion_matrix(y_true, y_pred, labels=list(range(num_classes)))
    save_confusion_matrix(cm, target_names, "confusion_matrix.png", normalized=False)
    save_confusion_matrix(cm, target_names, "normalized_confusion_matrix.png", normalized=True)

    errors_df = test_preds_df[
        test_preds_df["Actual Category"] != test_preds_df["Predicted Category"]
    ].copy()
    errors_df.to_csv(OUTPUT_DIR / "wrong_predictions.csv", index=False)

    sample_path = test_df.sample(1, random_state=SEED)["path"].iloc[0]
    prediction_results = predict_image(
        sample_path,
        model,
        eval_transform,
        idx_to_label,
        device,
        top_k=5,
    )
    prediction_results.to_csv(OUTPUT_DIR / "sample_prediction.csv", index=False)

    similar_results = find_similar_images(
        sample_path,
        df,
        model,
        eval_transform,
        device,
        top_k=8,
    )
    similar_results.to_csv(OUTPUT_DIR / "similar_images.csv", index=False)

    final_summary = {
        "project": "Lady Purse Image Classification",
        "model": "Fine-tuned ResNet18",
        "test_accuracy": float(test_acc),
        "test_macro_f1": float(test_macro_f1),
        "total_images": int(len(df)),
        "number_of_classes": int(num_classes),
        "classes": target_names,
        "checkpoint_path": str(checkpoint_path),
        "main_strength": "Strong performance on Bottega Veneta Mini Jodie and improved performance across designer purse categories.",
        "main_weakness": "Negative class has the lowest recall because it contains many visually different bag types.",
    }

    with open(OUTPUT_DIR / "final_project_summary.json", "w") as f:
        json.dump(final_summary, f, indent=2)

    print("\nSaved outputs to:", OUTPUT_DIR)
    print(final_summary)


if __name__ == "__main__":
    main()
