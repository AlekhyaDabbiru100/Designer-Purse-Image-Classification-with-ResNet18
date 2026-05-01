import json
import pandas as pd
import torch

from config import (
    SEED,
    DATA_DIR,
    OUTPUT_DIR,
    BASE_EPOCHS,
    BASE_LR,
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
    make_class_weighted_loss,
    run_epoch,
)


def main():
    set_seed(SEED)
    device = get_device()

    print("Device:", device)
    print("DATA_DIR:", DATA_DIR)
    print("OUTPUT_DIR:", OUTPUT_DIR)

    df, labels, label_to_idx, idx_to_label = build_manifest(DATA_DIR)
    num_classes = len(labels)
    df_split, train_df, val_df, test_df = stratified_split(df, seed=SEED)

    print("Total images:", len(df))
    print("Number of classes:", num_classes)
    print("Training images:", len(train_df))
    print("Validation images:", len(val_df))
    print("Test images:", len(test_df))

    print("\nClass counts:")
    print(df["display_label"].value_counts())

    split_table = pd.crosstab(df_split["display_label"], df_split["split"])
    split_table = split_table.rename(columns={
        "train": "Training",
        "val": "Validation",
        "test": "Test",
    })
    print("\nSplit table:")
    print(split_table)

    df_split.to_csv(OUTPUT_DIR / "manifest_with_splits.csv", index=False)

    with open(OUTPUT_DIR / "label_mapping.json", "w") as f:
        json.dump(
            {
                "label_to_idx": label_to_idx,
                "idx_to_label": idx_to_label,
                "display_name_map": DISPLAY_NAME_MAP,
            },
            f,
            indent=2,
        )

    train_loader, val_loader, _ = build_dataloaders(train_df, val_df, test_df)

    model = build_model(num_classes, freeze_backbone=True).to(device)

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    print("Total parameters:", total_params)
    print("Trainable parameters:", trainable_params)

    criterion = make_class_weighted_loss(train_df, num_classes, device)

    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=BASE_LR,
        weight_decay=1e-4,
    )

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="max",
        factor=0.5,
        patience=2,
    )

    best_val_f1 = -1
    history = []

    for epoch in range(1, BASE_EPOCHS + 1):
        print(f"\nEpoch {epoch}/{BASE_EPOCHS}")

        train_loss, train_acc, train_f1 = run_epoch(
            model,
            train_loader,
            criterion,
            device,
            optimizer=optimizer,
        )

        val_loss, val_acc, val_f1 = run_epoch(
            model,
            val_loader,
            criterion,
            device,
            optimizer=None,
        )

        scheduler.step(val_f1)

        row = {
            "epoch": epoch,
            "train_loss": train_loss,
            "train_acc": train_acc,
            "train_macro_f1": train_f1,
            "val_loss": val_loss,
            "val_acc": val_acc,
            "val_macro_f1": val_f1,
        }

        history.append(row)

        print(
            f"train_loss={train_loss:.4f} | "
            f"train_acc={train_acc:.4f} | "
            f"train_f1={train_f1:.4f} | "
            f"val_loss={val_loss:.4f} | "
            f"val_acc={val_acc:.4f} | "
            f"val_f1={val_f1:.4f}"
        )

        if val_f1 > best_val_f1:
            best_val_f1 = val_f1

            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "label_to_idx": label_to_idx,
                    "idx_to_label": idx_to_label,
                    "display_name_map": DISPLAY_NAME_MAP,
                    "img_size": 224,
                    "freeze_backbone": True,
                    "num_classes": num_classes,
                },
                BASE_CHECKPOINT_PATH,
            )

            print("Saved best baseline model.")

    history_df = pd.DataFrame(history)
    history_df.to_csv(OUTPUT_DIR / "training_history.csv", index=False)

    print("\nBest validation macro F1:", round(best_val_f1, 4))
    print("Saved training history to:", OUTPUT_DIR / "training_history.csv")
    print("Saved checkpoint to:", BASE_CHECKPOINT_PATH)


if __name__ == "__main__":
    main()
