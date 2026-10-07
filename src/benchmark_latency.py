"""Benchmark latensi & throughput inferensi untuk model FP32, Dynamic INT8, dan Static INT8.

Ini melengkapi metrik akurasi/ukuran yang sudah dihitung `quantize.py` -- latensi/throughput
adalah salah satu motivasi utama kuantisasi (model lebih kecil semestinya lebih cepat), jadi
perlu diukur eksplisit, bukan diasumsikan dari ukuran file saja.

Metodologi singkat (dicatat di sini supaya bisa disalin ke bagian metodologi naskah):
- Warm-up sebanyak `LATENCY_N_WARMUP` run dibuang sebelum pengukuran, untuk menghindari bias
  cold-start (alokasi memori pertama kali, pemanasan cache CPU, dsb).
- `LATENCY_N_RUNS` run diukur per kombinasi (model, batch size) memakai `time.perf_counter()`
  membungkus pemanggilan `session.run(...)` saja (tidak termasuk tokenisasi), dilaporkan sebagai
  mean, std, serta persentil p50/p95/p99 (persentil lebih tahan outlier dibanding mean saja).
- Batch size 1 merepresentasikan skenario latensi request tunggal (mis. layanan interaktif);
  batch size lain (default 8) merepresentasikan throughput saat diproses berkelompok.
- Semua diukur pada **CPU** (`CPUExecutionProvider`) di mesin yang sama untuk ketiga model,
  supaya perbandingan antar model adil (apple-to-apple) -- ini BUKAN angka latensi produksi/GPU,
  dan perlu disebut eksplisit sebagai keterbatasan bila mesin evaluasi berbeda dari target deploy.

Jalankan lewat terminal (prasyarat: export_onnx.py & quantize.py sudah selesai):
    cd src && python benchmark_latency.py

Output -> artifacts/evaluation/latency_benchmark.json
"""
import json
import time

import numpy as np
import pandas as pd
from transformers import AutoTokenizer

import config as cfg
import onnx_utils as ou


def benchmark_one(model_path, tokenizer, texts, batch_size: int):
    session = ou.get_session(model_path)
    input_names = {i.name for i in session.get_inputs()}
    output_name = session.get_outputs()[0].name

    # Batch INPUT TETAP (bukan di-generate ulang tiap run) supaya perbedaan waktu murni dari
    # komputasi model, bukan variasi panjang teks antar run.
    batch_texts = (texts * ((batch_size // len(texts)) + 1))[:batch_size]
    enc = tokenizer(batch_texts, return_tensors="np", padding=True, truncation=True, max_length=cfg.MAX_LENGTH)
    ort_inputs = {k: v.astype(np.int64) for k, v in enc.items() if k in input_names}

    for _ in range(cfg.LATENCY_N_WARMUP):
        session.run([output_name], ort_inputs)

    times_ms = np.empty(cfg.LATENCY_N_RUNS)
    for i in range(cfg.LATENCY_N_RUNS):
        t0 = time.perf_counter()
        session.run([output_name], ort_inputs)
        times_ms[i] = (time.perf_counter() - t0) * 1000

    return {
        "batch_size": batch_size,
        "n_runs": cfg.LATENCY_N_RUNS,
        "mean_ms": float(times_ms.mean()),
        "std_ms": float(times_ms.std()),
        "p50_ms": float(np.percentile(times_ms, 50)),
        "p95_ms": float(np.percentile(times_ms, 95)),
        "p99_ms": float(np.percentile(times_ms, 99)),
        "throughput_samples_per_sec": float(batch_size / (times_ms.mean() / 1000)),
    }


def main():
    cfg.ensure_dirs()
    paths = {
        "FP32": cfg.ONNX_FP32_PATH,
        "Dynamic_INT8": cfg.ONNX_DYNAMIC_PATH,
        "Static_INT8": cfg.ONNX_STATIC_PATH,
    }
    missing = [str(p) for p in paths.values() if not p.exists()]
    if missing:
        raise FileNotFoundError(
            "Model ONNX berikut belum ada:\n" + "\n".join(f"  - {p}" for p in missing) +
            "\nJalankan export_onnx.py lalu quantize.py dulu."
        )

    tokenizer = AutoTokenizer.from_pretrained(str(cfg.ONNX_DIR))
    test_df = pd.read_csv(cfg.TEST_CSV)
    # Sampel teks TETAP (seed sama dengan tahap lain) supaya ketiga model diukur pada input
    # yang identik -- perbandingan latensi jadi murni soal model, bukan soal variasi input.
    sample_texts = test_df["text"].sample(
        n=min(max(cfg.LATENCY_BATCH_SIZES), len(test_df)), random_state=cfg.RANDOM_SEED
    ).tolist()

    print(f"Benchmark pada {cfg.LATENCY_N_RUNS} run (+{cfg.LATENCY_N_WARMUP} warm-up) per kombinasi "
          f"model x batch size, provider: {ou.get_session(cfg.ONNX_FP32_PATH).get_providers()}")

    results = {}
    for name, path in paths.items():
        print(f"\n=== {name} ===")
        results[name] = {}
        for bs in cfg.LATENCY_BATCH_SIZES:
            r = benchmark_one(path, tokenizer, sample_texts, bs)
            results[name][f"batch_{bs}"] = r
            print(f"  batch={bs:<3} mean={r['mean_ms']:7.2f} ms | p95={r['p95_ms']:7.2f} ms | "
                  f"throughput={r['throughput_samples_per_sec']:8.1f} sampel/detik")

    with open(cfg.LATENCY_JSON, "w") as f:
        json.dump(results, f, indent=2)

    # Ringkasan relatif terhadap FP32 (batch=1) -- angka yang biasanya jadi headline di paper
    base = results["FP32"]["batch_1"]["mean_ms"]
    print("\n=== Ringkasan (batch=1, relatif terhadap FP32) ===")
    for name in paths:
        mean_ms = results[name]["batch_1"]["mean_ms"]
        speedup = base / mean_ms
        print(f"  {name:<14}: {mean_ms:7.2f} ms  (speedup {speedup:.2f}x)")

    print(f"\nSelesai. Hasil -> {cfg.LATENCY_JSON}")


if __name__ == "__main__":
    main()
