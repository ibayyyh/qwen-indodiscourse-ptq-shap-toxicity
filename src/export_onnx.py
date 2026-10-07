"""Export model FP32 (PyTorch) ke ONNX dengan native torch.onnx.export(dynamo=True).

Qwen2ForSequenceClassification mengembalikan objek ModelOutput (bukan tensor polos), dan
FeaturesManager/optimum tidak stabil untuk arsitektur ini di percobaan awal -- solusinya
adalah wrapper nn.Module tipis + exporter dynamo bawaan PyTorch >= 2.5.

Proses BERAT — jalankan lewat terminal (prasyarat: train.py sudah selesai):
    cd src && python export_onnx.py

Output -> artifacts/onnx/model_fp32.onnx, entry "FP32" di artifacts/evaluation/results_summary.json
"""
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import onnx
from torch.export import Dim
from transformers import AutoModelForSequenceClassification, AutoTokenizer

import config as cfg
import onnx_utils as ou


class QwenForONNX(nn.Module):
    """Bungkus model HF supaya forward() mengembalikan tensor logits polos (dibutuhkan
    torch.onnx.export, yang tidak bisa menangani objek ModelOutput secara langsung)."""

    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, input_ids, attention_mask):
        return self.model(input_ids=input_ids, attention_mask=attention_mask).logits


def main():
    cfg.ensure_dirs()
    if not cfg.FP32_MODEL_DIR.exists():
        raise FileNotFoundError(f"Model FP32 belum ada di {cfg.FP32_MODEL_DIR}. Jalankan train.py dulu.")

    print("=== [1/5] Load model & tokenizer FP32 ===")
    # torch_dtype=torch.float32 WAJIB eksplisit: config bawaan Qwen/Qwen2.5-0.5B menyatakan
    # torch_dtype=bfloat16, jadi tanpa ini model (termasuk head klasifikasi barunya) ikut
    # ter-load sebagai bfloat16 -- diam-diam BUKAN float32 seperti yang seharusnya untuk
    # baseline "FP32" penelitian ini, dan bikin .numpy() di sanity check meledak
    # (`TypeError: Got unsupported ScalarType BFloat16`) karena numpy tidak punya tipe itu.
    model = AutoModelForSequenceClassification.from_pretrained(
        str(cfg.FP32_MODEL_DIR), torch_dtype=torch.float32
    )
    model = model.float()  # extra safety net, jaga-jaga ada submodule yang lolos dari torch_dtype di atas
    tokenizer = AutoTokenizer.from_pretrained(str(cfg.FP32_MODEL_DIR))
    model.eval()

    onnx_model = QwenForONNX(model)
    onnx_model.eval()

    print("=== [2/5] Export ke ONNX (torch.onnx.export, dynamo=True) ===")
    dummy_input_ids = torch.ones((1, cfg.MAX_LENGTH), dtype=torch.long)
    dummy_attention_mask = torch.ones((1, cfg.MAX_LENGTH), dtype=torch.long)
    batch_dim, seq_dim = Dim("batch"), Dim("sequence")

    torch.onnx.export(
        onnx_model,
        (dummy_input_ids, dummy_attention_mask),
        str(cfg.ONNX_FP32_PATH),
        input_names=["input_ids", "attention_mask"],
        output_names=["logits"],
        dynamic_shapes={
            "input_ids": {0: batch_dim, 1: seq_dim},
            "attention_mask": {0: batch_dim, 1: seq_dim},
        },
        opset_version=17,
        dynamo=True,
    )
    print(f"ONNX tersimpan di: {cfg.ONNX_FP32_PATH}")
    tokenizer.save_pretrained(str(cfg.ONNX_DIR))  # dibutuhkan quantize.py & shap_analysis.py

    print("\n=== [3/5] Validasi struktur model ONNX ===")
    onnx_checked = onnx.load(str(cfg.ONNX_FP32_PATH))
    onnx.checker.check_model(onnx_checked)
    print("Valid           : YES")
    print(f"Ukuran          : {ou.file_size_mb(cfg.ONNX_FP32_PATH):.2f} MB")
    print("Inputs          :", [i.name for i in onnx_checked.graph.input])
    print("Outputs         :", [o.name for o in onnx_checked.graph.output])

    print("\n=== [4/5] Sanity check: PyTorch vs ONNX Runtime (1 sampel test) ===")
    test_df = pd.read_csv(cfg.TEST_CSV)
    sample_text = test_df["text"].iloc[0]
    enc = tokenizer(
        [sample_text], return_tensors="pt", padding="max_length", truncation=True, max_length=cfg.MAX_LENGTH
    )
    with torch.no_grad():
        pt_logits = onnx_model(enc["input_ids"], enc["attention_mask"]).numpy()

    session = ou.get_session(cfg.ONNX_FP32_PATH)
    ort_logits = session.run(
        ["logits"],
        {
            "input_ids": enc["input_ids"].numpy().astype(np.int64),
            "attention_mask": enc["attention_mask"].numpy().astype(np.int64),
        },
    )[0]
    max_diff = float(np.max(np.abs(pt_logits - ort_logits)))
    print(f"PyTorch logits : {pt_logits}")
    print(f"ONNX logits    : {ort_logits}")
    print(f"Max abs diff   : {max_diff:.8f}")
    if max_diff > 1e-3:
        print("PERINGATAN: selisih lebih besar dari toleransi wajar (1e-3) — cek ulang wrapper/export.")

    print("\n=== [5/5] Evaluasi ONNX FP32 pada test set penuh ===")
    metrics, preds = ou.evaluate(cfg.ONNX_FP32_PATH, test_df["text"].tolist(), test_df["label"].tolist(), tokenizer)
    for k, v in metrics.items():
        print(f"  {k}: {v:.4f}")

    size_mb = ou.file_size_mb(cfg.ONNX_FP32_PATH)
    ou.update_results_json({"FP32": {"size_mb": size_mb, **metrics}})
    ou.update_predictions_csv("fp32_pred", preds, test_df)

    print(f"\nSelesai. Hasil -> {cfg.RESULTS_JSON}, {cfg.PREDICTIONS_CSV}")
    print("Lanjutkan dengan: python quantize.py")


if __name__ == "__main__":
    main()
