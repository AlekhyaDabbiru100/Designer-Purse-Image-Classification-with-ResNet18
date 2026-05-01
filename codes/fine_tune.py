import pandas as pd
import torch

from config import (
    SEED,
    DATA_DIR,
    OUTPUT_DIR,
    FINE_TUNE_EPOCHS,
    FINE_TUNE_LR,
    BASE_CHECKPOINT_PATH,
    FINE_TUNE_CHECKPOINT_PATH,
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

    df, labels, label_to_idx, idx_to_label = build_manifest(DATA_DIR)
    num_classes = len(labels)
    _, train_df, val_df, test_df = stratified_split(df, seed=SEED)
    train_loader, val_loader, _ = build_dataloaders(train_df, val_df, test_df)

    checkpoint = torch.load(BASE_CHECKPOINT_PATH, map_location=device)

    model = build_model(num_classes, freeze_backbone=True).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])

    for param in model.parameters():
        param.requires_grad = False

    for param in model.layer4.parameters():
        param.requires_grad = True

    for param in model.fc.parameters():
        param.requires_grad = True

    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in model.parameters())

    print("Total parameters:", total_params)
    print("Trainable parameters:", trainable_params)

    criterion = make_class_weighted_loss(train_df, num_classes, device)

    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=FINE_TUNE_LR,
        weight_decay=1e-4,
    )

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="max",
        factor=0.5,
        patience=1,
    )

    history_path = OUTPUT_DIR / "training_history.csv"

    if history_path.exists():
        base_history = pd.read_csv(history_path)
        best_val_f1 = base_history["val_macro_f1"].max()
    else:
        best_val_f1 = -1

    fine_tune_history = []

    print("Starting validation macro F1:", round(best_val_f1, 4))

    for epoch in range(1, FINE_TUNE_EPOCHS + 1):
        print(f"\nFine-tuning Epoch {epoch}/{FINE_TUNE_EPOCHS}")

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
            "fine_tune_epoch": epoch,
            "train_loss": train_loss,
            "train_acc": train_acc,
            "train_macro_f1": train_f1,
            "val_loss": val_loss,
            "val_acc": val_acc,
            "val_macro_f1": val_f1,
        }

        fine_tune_history.append(row)

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
                    "freeze_backbone": False,
                    "num_classes": num_classes,
                },
                FINE_TUNE_CHECKPOINT_PATH,
            )

            print("Saved better fine-tuned model.")

    fine_tune_history_df = pd.DataFrame(fine_tune_history)
    fine_tune_history_df.to_csv(OUTPUT_DIR / "fine_tune_history.csv", index=False)

    print("\nBest fine-tuned validation macro F1:", round(best_val_f1, 4))
    print("Saved fine-tune history to:", OUTPUT_DIR / "fine_tune_history.csv")
    print("Saved checkpoint to:", FINE_TUNE_CHECKPOINT_PATH)


if __name__ == "__main__":
    main()
