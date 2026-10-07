"""Interpretasi SHAP untuk model FP32, Dynamic INT8, dan Static INT8 + metrik kemiripan.

Proses BERAT (banyak forward pass per token yang di-mask) — jalankan lewat terminal
(prasyarat: export_onnx.py & quantize.py sudah selesai):
    cd src && python shap_analysis.py

Output -> artifacts/shap/ (pickle token-level, CSV kemiripan, plot PNG, HTML highlight).
"""
import pickle

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import shap
from transformers import AutoTokenizer
from scipy.stats import pearsonr, spearmanr

import config as cfg
import onnx_utils as ou


def truncate(text: str, max_words: int = cfg.SHAP_MAX_WORDS) -> str:
    words = text.split()
    return text if len(words) <= max_words else " ".join(words[:max_words])


def cosine_sim(a, b):
    denom = np.linalg.norm(a) * np.linalg.norm(b)
    return float(np.dot(a, b) / denom) if denom else np.nan


def top_k_jaccard(a, b, k=5):
    k = min(k, len(a))
    if k == 0:
        return np.nan
    top_a = set(np.argsort(-np.abs(a))[:k])
    top_b = set(np.argsort(-np.abs(b))[:k])
    return len(top_a & top_b) / len(top_a | top_b)


def safe_corr(fn, a, b):
    return fn(a, b)[0] if np.std(a) > 0 and np.std(b) > 0 else np.nan


def sample_texts(test_df: pd.DataFrame) -> pd.DataFrame:
    """Ambil campuran toxic & non-toxic (bukan random polos) supaya kedua kelas terwakili
    walau datanya timpang (~8% toxic).

    Sengaja TIDAK pakai groupby(...).apply(lambda g: g.sample(...)): di pandas >= 2.2,
    kolom yang dipakai untuk grouping dikeluarkan dari DataFrame `g` yang diterima lambda,
    sehingga `sample_df["label"]` di bawah akan KeyError. Iterasi manual di bawah ini aman
    di semua versi pandas karena tidak melalui .apply() sama sekali.
    """
    n_per_class = max(1, cfg.N_SHAP_SAMPLES // cfg.NUM_LABELS)
    parts = []
    for _, group in test_df.groupby("label"):
        parts.append(group.sample(n=min(n_per_class, len(group)), random_state=cfg.RANDOM_SEED))
    return pd.concat(parts).reset_index(drop=True)


def main():
    cfg.ensure_dirs()
    missing = [p for p in [cfg.ONNX_FP32_PATH, cfg.ONNX_DYNAMIC_PATH, cfg.ONNX_STATIC_PATH] if not p.exists()]
    if missing:
        raise FileNotFoundError(
            "Model ONNX berikut belum ada:\n" + "\n".join(f"  - {p}" for p in missing) +
            "\nJalankan export_onnx.py lalu quantize.py dulu."
        )

    tokenizer = AutoTokenizer.from_pretrained(str(cfg.ONNX_DIR))
    test_df = pd.read_csv(cfg.TEST_CSV)

    sample_df = sample_texts(test_df)
    texts = [truncate(t) for t in sample_df["text"].tolist()]
    true_labels = sample_df["label"].tolist()
    print(f"=== [1/4] Menjelaskan {len(texts)} sampel dengan SHAP (FP32, Dynamic, Static) ===")

    paths = {"fp32": cfg.ONNX_FP32_PATH, "dynamic": cfg.ONNX_DYNAMIC_PATH, "static": cfg.ONNX_STATIC_PATH}
    masker = shap.maskers.Text(tokenizer)

    shap_values, probs = {}, {}
    for name, path in paths.items():
        predict_fn = lambda t, p=path: ou.predict_proba(t, p, tokenizer)
        shap_values[name] = shap.Explainer(predict_fn, masker)(texts)
        probs[name] = predict_fn(texts)
        print(f"  [{name}] selesai")

    ref_class = np.argmax(probs["fp32"], axis=1)

    print("\n=== [2/4] Hitung metrik kemiripan SHAP terhadap FP32 ===")
    rows, token_level = [], []
    for i in range(len(texts)):
        c = int(ref_class[i])
        vecs = {n: np.asarray(shap_values[n][i, :, c].values, dtype=float) for n in paths}
        tokens = list(shap_values["fp32"][i, :, c].data)

        rows.append({
            "sample_idx": i,
            "true_label": cfg.LABEL_NAMES[true_labels[i]],
            "explained_class": cfg.LABEL_NAMES[c],
            "n_tokens": len(tokens),
            "cosine_fp32_vs_dynamic": cosine_sim(vecs["fp32"], vecs["dynamic"]),
            "cosine_fp32_vs_static": cosine_sim(vecs["fp32"], vecs["static"]),
            "pearson_fp32_vs_dynamic": safe_corr(pearsonr, vecs["fp32"], vecs["dynamic"]),
            "pearson_fp32_vs_static": safe_corr(pearsonr, vecs["fp32"], vecs["static"]),
            "spearman_fp32_vs_dynamic": safe_corr(spearmanr, vecs["fp32"], vecs["dynamic"]),
            "spearman_fp32_vs_static": safe_corr(spearmanr, vecs["fp32"], vecs["static"]),
            "jaccard_top5_fp32_vs_dynamic": top_k_jaccard(vecs["fp32"], vecs["dynamic"]),
            "jaccard_top5_fp32_vs_static": top_k_jaccard(vecs["fp32"], vecs["static"]),
        })
        token_level.append({
            "text": texts[i],
            "true_label": cfg.LABEL_NAMES[true_labels[i]],
            "explained_class": cfg.LABEL_NAMES[c],
            "tokens": tokens,
            "shap_fp32": vecs["fp32"], "shap_dynamic": vecs["dynamic"], "shap_static": vecs["static"],
        })

    sim_df = pd.DataFrame(rows).round(4)
    sim_df.to_csv(cfg.SHAP_SIMILARITY_CSV, index=False)

    summary_cols = [c for c in sim_df.columns if c.startswith(("cosine", "pearson", "spearman", "jaccard"))]
    summary_df = sim_df[summary_cols].agg(["mean", "std"]).T.round(4)
    summary_df.to_csv(cfg.SHAP_SUMMARY_CSV)
    print(summary_df)

    with open(cfg.SHAP_TOKENS_PKL, "wb") as f:
        pickle.dump(token_level, f)

    print("\n=== [3/4] Simpan plot global importance (PNG) ===")
    for name in paths:
        shap.plots.bar(shap_values[name][:, :, 1].mean(0), show=False)
        plt.title(f"Global token importance (toxic) - {name}")
        plt.tight_layout()
        plt.savefig(cfg.SHAP_PLOTS_DIR / f"global_importance_{name}.png", dpi=120)
        plt.close()

    print("=== [4/4] Simpan contoh highlight token (HTML) ===")
    for name in paths:
        for i in range(min(3, len(texts))):
            html = shap.plots.text(shap_values[name][i, :, int(ref_class[i])], display=False)
            if html:
                with open(cfg.SHAP_HTML_DIR / f"{name}_sample{i}.html", "w", encoding="utf-8") as f:
                    f.write(html)

    print(f"\nSelesai. Hasil -> {cfg.SHAP_DIR}")
    print("Semua tahap komputasi berat selesai. Buka notebooks/01_analisis.ipynb untuk analisis akhir.")


if __name__ == "__main__":
    main()
