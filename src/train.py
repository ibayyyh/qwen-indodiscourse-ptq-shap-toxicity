"""Fine-tuning Qwen2.5-0.5B untuk klasifikasi toksisitas (IndoToxic2024).

Proses BERAT — jalankan lewat terminal, BUKAN di notebook:
    cd src && python train.py
    (atau dari root repo: python src/train.py)

Output -> artifacts/fp32/final/ (model + tokenizer), artifacts/evaluation/ (split data + metrik).
"""
import json
import sys

import numpy as np
import torch
from datasets import Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer,
)
from sklearn.metrics import accuracy_score, precision_recall_fscore_support

import config as cfg
from data_utils import load_and_preprocess


def print_environment():
    print("=== ENVIRONMENT ===")
    print("Python :", sys.version.split()[0])
    print("PyTorch:", torch.__version__)
    print("CUDA   :", torch.cuda.is_available())
    if torch.cuda.is_available():
        print("GPU    :", torch.cuda.get_device_name(0))
        print("CUDA v :", torch.version.cuda)
    else:
        print("Tidak ada GPU terdeteksi — training akan berjalan di CPU (sangat lambat untuk 0.5B model).")


def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=-1)
    acc = accuracy_score(labels, preds)
    prec, rec, f1, _ = precision_recall_fscore_support(
        labels, preds, average="binary", pos_label=1, zero_division=0
    )
    return {"accuracy": acc, "precision": prec, "recall": rec, "f1": f1}


def main():
    cfg.ensure_dirs()
    print_environment()

    print("\n=== [1/6] Load & preprocessing dataset ===")
    train_df, val_df, test_df = load_and_preprocess()
    train_df.to_csv(cfg.TRAIN_CSV, index=False)
    val_df.to_csv(cfg.VAL_CSV, index=False)
    test_df.to_csv(cfg.TEST_CSV, index=False)
    print(f"Split disimpan di: {cfg.SPLIT_DIR}")

    print("\n=== [2/6] Tokenisasi ===")
    tokenizer = AutoTokenizer.from_pretrained(cfg.MODEL_NAME)
    print("Tokenizer :", cfg.MODEL_NAME)
    print("Pad token :", tokenizer.pad_token)

    def tokenize_batch(batch):
        return tokenizer(batch["text"], padding="max_length", truncation=True, max_length=cfg.MAX_LENGTH)

    train_ds = Dataset.from_pandas(train_df[["text", "label"]].reset_index(drop=True)).map(tokenize_batch, batched=True)
    val_ds = Dataset.from_pandas(val_df[["text", "label"]].reset_index(drop=True)).map(tokenize_batch, batched=True)
    test_ds = Dataset.from_pandas(test_df[["text", "label"]].reset_index(drop=True)).map(tokenize_batch, batched=True)

    print("\n=== [3/6] Load model & siapkan Trainer ===")
    model = AutoModelForSequenceClassification.from_pretrained(
        cfg.MODEL_NAME, num_labels=cfg.NUM_LABELS, torch_dtype=torch.float32
    )
    # Wajib eksplisit: config bawaan Qwen/Qwen2.5-0.5B menyatakan torch_dtype=bfloat16,
    # jadi tanpa ini baseline "FP32" kita diam-diam ke-load sebagai bfloat16.
    model.config.pad_token_id = tokenizer.pad_token_id
    model.gradient_checkpointing_enable()
    model.config.use_cache = False
    print("Parameter model:", sum(p.numel() for p in model.parameters()))

    training_args = TrainingArguments(
        output_dir=str(cfg.FP32_TRAINER_DIR),
        num_train_epochs=cfg.NUM_TRAIN_EPOCHS,
        per_device_train_batch_size=cfg.TRAIN_BATCH_SIZE,
        per_device_eval_batch_size=cfg.EVAL_BATCH_SIZE,
        gradient_accumulation_steps=cfg.GRAD_ACCUM_STEPS,
        learning_rate=cfg.LEARNING_RATE,
        weight_decay=cfg.WEIGHT_DECAY,
        eval_strategy="epoch",
        save_strategy="epoch",
        logging_strategy="steps",
        logging_steps=100,
        load_best_model_at_end=True,
        metric_for_best_model="f1",
        greater_is_better=True,
        fp16=False,
        bf16=False,
        report_to="none",
        seed=cfg.RANDOM_SEED,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        processing_class=tokenizer,
        compute_metrics=compute_metrics,
    )

    print("\n=== [4/6] Training ===")
    train_result = trainer.train()
    print("Training selesai.", train_result)

    print("\n=== [5/6] Evaluasi (val + test) ===")
    val_result = trainer.evaluate(metric_key_prefix="val")
    test_result = trainer.evaluate(eval_dataset=test_ds, metric_key_prefix="test")
    print("Val :", val_result)
    print("Test:", test_result)

    metrics_out = {k: float(v) for k, v in {**val_result, **test_result}.items() if isinstance(v, (int, float))}
    with open(cfg.FP32_PYTORCH_METRICS_JSON, "w") as f:
        json.dump(metrics_out, f, indent=2)

    print("\n=== [6/6] Simpan model FP32 ===")
    trainer.save_model(str(cfg.FP32_MODEL_DIR))
    tokenizer.save_pretrained(str(cfg.FP32_MODEL_DIR))
    print(f"Model + tokenizer tersimpan di: {cfg.FP32_MODEL_DIR}")
    print(f"Metrik val/test tersimpan di  : {cfg.FP32_PYTORCH_METRICS_JSON}")
    print("\nSelesai. Lanjutkan dengan: python export_onnx.py")


if __name__ == "__main__":
    main()
