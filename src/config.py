"""Konfigurasi tunggal untuk seluruh pipeline (script terminal & notebook).

Semua path dihitung relatif terhadap lokasi file ini (`Path(__file__)`), BUKAN terhadap
current working directory. Artinya script ini tetap benar dipanggil dari mana saja:
    python src/train.py          (dari root repo)
    cd src && python train.py    (dari dalam src/)
    dari dalam notebooks/01_analisis.ipynb
"""
from pathlib import Path

# --- Path dasar repo ---
REPO_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = REPO_ROOT / "data"
ARTIFACTS_DIR = REPO_ROOT / "artifacts"

RAW_CSV_PATH = DATA_DIR / "indotoxic2024_annotated_data_v2_final.csv"

# --- Sub-folder artifact (terpisah dari notebook, lihat README) ---
FP32_DIR = ARTIFACTS_DIR / "fp32"
FP32_MODEL_DIR = FP32_DIR / "final"          # model + tokenizer hasil fine-tuning
FP32_TRAINER_DIR = FP32_DIR / "trainer_run"  # checkpoint mentah Trainer

ONNX_DIR = ARTIFACTS_DIR / "onnx"
ONNX_FP32_PATH = ONNX_DIR / "model_fp32.onnx"

QUANTIZED_DIR = ARTIFACTS_DIR / "quantized"
ONNX_DYNAMIC_PATH = QUANTIZED_DIR / "model_dynamic_int8.onnx"
ONNX_STATIC_PREP_PATH = QUANTIZED_DIR / "model_fp32_preprocessed.onnx"  # hasil quant_pre_process
ONNX_STATIC_PATH = QUANTIZED_DIR / "model_static_int8.onnx"

EVAL_DIR = ARTIFACTS_DIR / "evaluation"
SPLIT_DIR = EVAL_DIR / "splits"              # train/val/test csv hasil preprocessing
TRAIN_CSV = SPLIT_DIR / "train.csv"
VAL_CSV = SPLIT_DIR / "val.csv"
TEST_CSV = SPLIT_DIR / "test.csv"

FP32_PYTORCH_METRICS_JSON = EVAL_DIR / "fp32_pytorch_metrics.json"  # hasil train.py (val + test)
RESULTS_JSON = EVAL_DIR / "results_summary.json"      # ukuran & metrik FP32/Dynamic/Static (ONNX)
PREDICTIONS_CSV = EVAL_DIR / "predictions_test.csv"    # prediksi ketiga model pada test set
LATENCY_JSON = EVAL_DIR / "latency_benchmark.json"     # latensi & throughput inferensi ketiga model

SHAP_DIR = ARTIFACTS_DIR / "shap"
SHAP_TOKENS_PKL = SHAP_DIR / "shap_token_level.pkl"
SHAP_SIMILARITY_CSV = SHAP_DIR / "shap_similarity_per_sample.csv"
SHAP_SUMMARY_CSV = SHAP_DIR / "shap_similarity_summary.csv"
SHAP_PLOTS_DIR = SHAP_DIR / "plots"
SHAP_HTML_DIR = SHAP_DIR / "html"

# --- Dataset & label ---
TARGET_COLUMN = "toxicity"           # kolom anotasi yang dipakai sebagai target (lihat data_utils.py)
LABEL_NAMES = {0: "non_toxic", 1: "toxic"}
NUM_LABELS = len(LABEL_NAMES)
RANDOM_SEED = 42

# --- Split (mengikuti metodologi awal: 80% train, 10% val, 10% test, stratified) ---
TEST_SIZE = 0.20        # porsi disisihkan dari total (dibagi lagi jadi val & test)
VAL_TEST_SPLIT = 0.50   # separuh dari TEST_SIZE untuk test, separuh untuk val -> 80/10/10

# Mode cepat untuk uji coba SELURUH pipeline (train -> export -> quantize -> shap) dalam
# hitungan menit, bukan jam -- berguna untuk memastikan tidak ada error dulu sebelum run penuh.
# Contoh: DEV_SAMPLE_SIZE = 300. HASILNYA TIDAK VALID UNTUK SKRIPSI -- kembalikan ke None
# sebelum menjalankan pipeline final. Minimal ~200 (supaya kelas minoritas 'toxic' ~8% tetap
# ada cukup sampel di tiap split setelah dibagi train/val/test).
DEV_SAMPLE_SIZE = None

# --- Model & tokenisasi ---
MODEL_NAME = "Qwen/Qwen2.5-0.5B"
MAX_LENGTH = 256

# --- Fine-tuning (disesuaikan untuk GPU dengan VRAM terbatas) ---
NUM_TRAIN_EPOCHS = 3
TRAIN_BATCH_SIZE = 1
EVAL_BATCH_SIZE = 1
GRAD_ACCUM_STEPS = 16     # effective batch size = 16
LEARNING_RATE = 2e-5
WEIGHT_DECAY = 0.01

# --- Kuantisasi statis (PTQ Static) ---
CALIBRATION_SIZE = 256

# --- Benchmark latensi inferensi ---
LATENCY_N_WARMUP = 10          # jumlah run dibuang di awal (cold-start: alokasi memori, cache CPU)
LATENCY_N_RUNS = 100           # jumlah run yang diukur & dirata-ratakan per (model, batch size)
LATENCY_BATCH_SIZES = [1, 8]   # batch=1 -> latensi request tunggal; batch=8 -> throughput

# --- SHAP ---
N_SHAP_SAMPLES = 12
SHAP_MAX_WORDS = 40


def ensure_dirs():
    """Buat semua folder artifact yang dibutuhkan (aman dipanggil berulang kali)."""
    for d in [
        DATA_DIR, ARTIFACTS_DIR,
        FP32_DIR, FP32_MODEL_DIR, FP32_TRAINER_DIR,
        ONNX_DIR, QUANTIZED_DIR,
        EVAL_DIR, SPLIT_DIR,
        SHAP_DIR, SHAP_PLOTS_DIR, SHAP_HTML_DIR,
    ]:
        d.mkdir(parents=True, exist_ok=True)
