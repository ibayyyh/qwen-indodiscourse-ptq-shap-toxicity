"""Kuantisasi Post-Training: Dynamic INT8 & Static INT8.

Proses BERAT — jalankan lewat terminal (prasyarat: export_onnx.py sudah selesai):
    cd src && python quantize.py

Output -> artifacts/quantized/{model_dynamic_int8.onnx, model_static_int8.onnx},
entry "Dynamic_INT8" & "Static_INT8" di artifacts/evaluation/results_summary.json
"""
import numpy as np
import pandas as pd
import onnxruntime as ort
from transformers import AutoTokenizer
from onnxruntime.quantization import (
    quantize_dynamic,
    quantize_static,
    CalibrationDataReader,
    QuantType,
    QuantFormat,
    CalibrationMethod,
)
from onnxruntime.quantization.shape_inference import quant_pre_process

import config as cfg
import onnx_utils as ou


class TestCalibrationDataReader(CalibrationDataReader):
    """Suplai batch kalibrasi satu per satu ke quantize_static (dibutuhkan untuk
    menghitung rentang aktivasi sebelum dikuantisasi ke INT8)."""

    def __init__(self, samples):
        self.samples = samples
        self.index = 0

    def get_next(self):
        if self.index >= len(self.samples):
            return None
        sample = self.samples[self.index]
        self.index += 1
        return sample

    def rewind(self):
        self.index = 0


def build_calibration_reader(train_df: pd.DataFrame, tokenizer) -> TestCalibrationDataReader:
    calib_texts = (
        train_df["text"]
        .sample(n=min(cfg.CALIBRATION_SIZE, len(train_df)), random_state=cfg.RANDOM_SEED)
        .tolist()
    )
    session = ort.InferenceSession(str(cfg.ONNX_FP32_PATH), providers=["CPUExecutionProvider"])
    input_names = [i.name for i in session.get_inputs()]
    del session

    samples = []
    for text in calib_texts:
        enc = tokenizer(text, return_tensors="np", padding="max_length", truncation=True, max_length=cfg.MAX_LENGTH)
        samples.append({k: enc[k].astype(np.int64) for k in input_names if k in enc})
    return TestCalibrationDataReader(samples)


def run_dynamic(test_df, tokenizer):
    print("=== PTQ DYNAMIC (INT8) ===")
    quantize_dynamic(
        model_input=str(cfg.ONNX_FP32_PATH),
        model_output=str(cfg.ONNX_DYNAMIC_PATH),
        weight_type=QuantType.QInt8,
    )
    print(f"Model tersimpan: {cfg.ONNX_DYNAMIC_PATH}")

    metrics, preds = ou.evaluate(cfg.ONNX_DYNAMIC_PATH, test_df["text"].tolist(), test_df["label"].tolist(), tokenizer)
    size_mb = ou.file_size_mb(cfg.ONNX_DYNAMIC_PATH)
    print(f"Ukuran: {size_mb:.2f} MB | {metrics}")

    ou.update_results_json({"Dynamic_INT8": {"size_mb": size_mb, **metrics}})
    ou.update_predictions_csv("dynamic_pred", preds, test_df)


def run_static(test_df, train_df, tokenizer):
    print("\n=== PTQ STATIC (INT8) ===")
    print("[1/3] Pre-processing (symbolic shape inference)...")
    try:
        quant_pre_process(
            input_model_path=str(cfg.ONNX_FP32_PATH),
            output_model_path=str(cfg.ONNX_STATIC_PREP_PATH),
            skip_symbolic_shape=False,
        )
    except Exception as e:
        # Beberapa graf hasil torch.onnx.export(dynamo=True) (mis. pola Range dinamis untuk
        # position_ids) belum sepenuhnya didukung SymbolicShapeInference bawaan onnxruntime.
        # Fallback: skip symbolic shape inference, tetap lanjut tanpa optimasi shape tsb
        # (quantize_static tetap bisa jalan, hanya sedikit kurang optimal).
        print(f"  Symbolic shape inference gagal ({type(e).__name__}: {e}); mencoba tanpa symbolic shape inference...")
        quant_pre_process(
            input_model_path=str(cfg.ONNX_FP32_PATH),
            output_model_path=str(cfg.ONNX_STATIC_PREP_PATH),
            skip_symbolic_shape=True,
        )

    print(f"[2/3] Kalibrasi ({cfg.CALIBRATION_SIZE} sampel dari train set)...")
    calib_reader = build_calibration_reader(train_df, tokenizer)

    print("[3/3] Quantize static (QDQ, INT8 weight + activation)...")
    quantize_static(
        model_input=str(cfg.ONNX_STATIC_PREP_PATH),
        model_output=str(cfg.ONNX_STATIC_PATH),
        calibration_data_reader=calib_reader,
        quant_format=QuantFormat.QDQ,
        activation_type=QuantType.QInt8,
        weight_type=QuantType.QInt8,
        calibrate_method=CalibrationMethod.MinMax,
    )
    print(f"Model tersimpan: {cfg.ONNX_STATIC_PATH}")

    metrics, preds = ou.evaluate(cfg.ONNX_STATIC_PATH, test_df["text"].tolist(), test_df["label"].tolist(), tokenizer)
    size_mb = ou.file_size_mb(cfg.ONNX_STATIC_PATH)
    print(f"Ukuran: {size_mb:.2f} MB | {metrics}")

    ou.update_results_json({"Static_INT8": {"size_mb": size_mb, **metrics}})
    ou.update_predictions_csv("static_pred", preds, test_df)


def main():
    cfg.ensure_dirs()
    if not cfg.ONNX_FP32_PATH.exists():
        raise FileNotFoundError(f"ONNX FP32 belum ada di {cfg.ONNX_FP32_PATH}. Jalankan export_onnx.py dulu.")

    tokenizer = AutoTokenizer.from_pretrained(str(cfg.FP32_MODEL_DIR))
    test_df = pd.read_csv(cfg.TEST_CSV)
    train_df = pd.read_csv(cfg.TRAIN_CSV)

    run_dynamic(test_df, tokenizer)
    run_static(test_df, train_df, tokenizer)

    print("\n=== RINGKASAN (FP32 vs Dynamic vs Static) ===")
    import json
    with open(cfg.RESULTS_JSON) as f:
        results = json.load(f)
    print(pd.DataFrame(results).T)

    print(f"\nSelesai. Hasil lengkap -> {cfg.RESULTS_JSON}, {cfg.PREDICTIONS_CSV}")
    print("Lanjutkan dengan: python shap_analysis.py")


if __name__ == "__main__":
    main()
