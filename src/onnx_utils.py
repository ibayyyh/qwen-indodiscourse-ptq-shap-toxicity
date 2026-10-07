"""Fungsi prediksi & evaluasi berbasis ONNX Runtime.

Dipakai bersama oleh export_onnx.py, quantize.py, shap_analysis.py, dan notebook —
supaya FP32/Dynamic/Static dievaluasi dengan cara yang persis sama (apple-to-apple).
"""
import json

import numpy as np
import pandas as pd
import onnxruntime as ort
from sklearn.metrics import accuracy_score, precision_recall_fscore_support

import config as cfg

_SESSION_CACHE = {}


def get_session(model_path) -> ort.InferenceSession:
    """Buat (atau ambil dari cache) ONNX Runtime session. Coba CUDA, fallback ke CPU otomatis."""
    key = str(model_path)
    if key not in _SESSION_CACHE:
        _SESSION_CACHE[key] = ort.InferenceSession(
            key, providers=["CUDAExecutionProvider", "CPUExecutionProvider"]
        )
    return _SESSION_CACHE[key]


def predict_proba(texts, model_path, tokenizer, batch_size: int = 8) -> np.ndarray:
    """Softmax probability, shape (n_texts, NUM_LABELS). `model_path` = file .onnx."""
    if isinstance(texts, str):
        texts = [texts]
    texts = [str(t) for t in texts]

    session = get_session(model_path)
    input_names = {i.name for i in session.get_inputs()}
    output_name = session.get_outputs()[0].name

    probs = []
    for i in range(0, len(texts), batch_size):
        batch = texts[i:i + batch_size]
        enc = tokenizer(batch, return_tensors="np", padding=True, truncation=True, max_length=cfg.MAX_LENGTH)
        ort_inputs = {k: v.astype(np.int64) for k, v in enc.items() if k in input_names}
        logits = session.run([output_name], ort_inputs)[0]
        exp = np.exp(logits - logits.max(axis=1, keepdims=True))
        probs.append(exp / exp.sum(axis=1, keepdims=True))
    return np.vstack(probs)


def predict_class(texts, model_path, tokenizer, batch_size: int = 8) -> np.ndarray:
    return np.argmax(predict_proba(texts, model_path, tokenizer, batch_size), axis=1)


def evaluate(model_path, texts, labels, tokenizer, batch_size: int = 8):
    """Accuracy/precision/recall/F1 (average='binary', positive class = 1/toxic)."""
    preds = predict_class(texts, model_path, tokenizer, batch_size)
    labels = np.asarray(labels)
    acc = accuracy_score(labels, preds)
    prec, rec, f1, _ = precision_recall_fscore_support(
        labels, preds, average="binary", pos_label=1, zero_division=0
    )
    metrics = {"accuracy": float(acc), "precision": float(prec), "recall": float(rec), "f1": float(f1)}
    return metrics, preds


def file_size_mb(path) -> float:
    """Total ukuran model di disk (MB).

    `torch.onnx.export(..., dynamo=True)` defaultnya `external_data=True`: bobot model
    disimpan di file terpisah (mis. `model_fp32.onnx.data`) di folder yang sama dengan
    `.onnx`-nya, bukan di dalam file `.onnx` itu sendiri. Supaya perbandingan ukuran FP32
    vs Dynamic vs Static tetap benar, kita jumlahkan semua file yang namanya diawali nama
    file model tersebut (menangkap baik kasus single-file maupun external-data).
    """
    from pathlib import Path
    path = Path(path)
    return sum(p.stat().st_size for p in path.parent.glob(f"{path.name}*") if p.is_file()) / (1024 ** 2)


def update_results_json(entry: dict) -> dict:
    """Gabungkan `entry` (mis. {"FP32": {...}}) ke RESULTS_JSON tanpa menghapus entry lain
    yang sudah ditulis script sebelumnya (export_onnx.py / quantize.py bisa dijalankan
    berulang & independen)."""
    results = {}
    if cfg.RESULTS_JSON.exists():
        with open(cfg.RESULTS_JSON) as f:
            results = json.load(f)
    results.update(entry)
    with open(cfg.RESULTS_JSON, "w") as f:
        json.dump(results, f, indent=2)
    return results


def update_predictions_csv(column_name: str, preds, test_df: pd.DataFrame) -> pd.DataFrame:
    """Tambah/replace satu kolom prediksi di PREDICTIONS_CSV (dibuat jika belum ada)."""
    if cfg.PREDICTIONS_CSV.exists():
        pred_df = pd.read_csv(cfg.PREDICTIONS_CSV)
    else:
        pred_df = test_df[["text_id", "text", "label", "label_name"]].reset_index(drop=True)
    pred_df[column_name] = preds
    pred_df.to_csv(cfg.PREDICTIONS_CSV, index=False)
    return pred_df
