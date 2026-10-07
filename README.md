# Analisis Pengaruh Kuantisasi terhadap Explainable AI (XAI)
### Studi Kasus: Fine-tuning Qwen2.5-0.5B untuk Klasifikasi Toksisitas pada IndoToxic2024

Repo tugas akhir ini menguji apakah kuantisasi post-training (**PTQ Dynamic** & **PTQ Static**, INT8)
pada model bahasa yang telah di-fine-tune mempengaruhi **(1) performa klasifikasi** dan
**(2) konsistensi interpretasi SHAP** dibanding model asal (FP32).

**Notebook utama**: [`notebooks/01_analisis.ipynb`](notebooks/01_analisis.ipynb) — inti tugas akhir ini.
Semua proses berat (training, export, kuantisasi, SHAP) dijalankan lewat script di `src/` dari
terminal, **bukan** di notebook, supaya notebook tetap ringan, cepat dibuka, dan tidak berisiko
force-close. Hasil tiap script disimpan sebagai *artifact* di folder `artifacts/` (terpisah dari
notebook); notebook hanya **membaca** artifact tersebut untuk evaluasi, analisis, dan visualisasi.

---

## Daftar Isi

1. [Struktur Repo](#struktur-repo)
2. [Dataset](#dataset)
3. [Spesifikasi Hardware & OS](#spesifikasi-hardware--os)
4. [Instalasi Environment](#instalasi-environment)
5. [Alur Eksperimen](#alur-eksperimen)
6. [Menjalankan Script Terminal](#menjalankan-script-terminal)
7. [Menjalankan Notebook](#menjalankan-notebook)
8. [Lokasi Artifact](#lokasi-artifact)
9. [Konfigurasi](#konfigurasi)
10. [Kustomisasi & Pertanyaan Umum (FAQ)](#kustomisasi--pertanyaan-umum-faq)
11. [Troubleshooting](#troubleshooting)
12. [Langkah Reproduksi Lengkap (Ringkas)](#langkah-reproduksi-lengkap-ringkas)
13. [Lisensi & Sitasi](#lisensi--sitasi)

---

## Struktur Repo

```
qwen-indodiscourse-quant-xai/
├── README.md
├── requirements.txt
├── .gitignore
├── data/
│   └── indotoxic2024_annotated_data_v2_final.csv   # dataset mentah IndoToxic2024
├── src/                            # SEMUA proses berat — dijalankan lewat terminal
│   ├── config.py                      # satu-satunya tempat mengubah konfigurasi/hyperparameter
│   ├── data_utils.py                   # load & preprocessing (majority vote, split 80/10/10)
│   ├── onnx_utils.py                    # helper inference/evaluasi ONNX (dipakai bersama)
│   ├── train.py                          # [1] fine-tuning Qwen2.5-0.5B    -> artifacts/fp32/
│   ├── export_onnx.py                     # [2] export ke ONNX             -> artifacts/onnx/
│   ├── quantize.py                         # [3] PTQ Dynamic & Static      -> artifacts/quantized/
│   ├── benchmark_latency.py                 # [4] latensi & throughput     -> artifacts/evaluation/
│   └── shap_analysis.py                      # [5] interpretasi SHAP        -> artifacts/shap/
├── notebooks/
│   └── 01_analisis.ipynb            # SATU-SATUNYA notebook: load artifact -> analisis & visualisasi
└── artifacts/                       # terpisah dari notebooks/ — semua output proses berat
    ├── fp32/final/                      # model + tokenizer PyTorch hasil fine-tuning
    ├── onnx/                             # model ONNX FP32 + tokenizer
    ├── quantized/                         # model ONNX Dynamic INT8 & Static INT8
    ├── evaluation/                         # split data (CSV), metrik (JSON), prediksi (CSV)
    └── shap/                                # hasil SHAP: pickle, CSV kemiripan, plot PNG, HTML
```

Path di `src/config.py` dihitung otomatis dari lokasi file (`Path(__file__)`), **bukan** path absolut
atau string relatif manual (`"../..."`) — sehingga repo ini bisa dijalankan dari direktori mana pun,
di komputer siapa pun, tanpa mengubah apa pun secara manual.

## Dataset

`data/indotoxic2024_annotated_data_v2_final.csv` (~13 MB, 28.448 baris) — data mentah IndoToxic2024:
teks media sosial/berita berbahasa Indonesia terkait Pemilu 2024, tiap teks dianotasi oleh beberapa
annotator untuk toksisitas, polarisasi, dan kategori lain (disimpan sebagai string list, mis.
`"['0', '1']"`).

`src/data_utils.py` melakukan:
1. **Parse** kolom anotasi (`ast.literal_eval`).
2. **Agregasi label** lewat *strict majority vote* pada kolom `toxicity`.
3. **Buang teks dengan suara seri** (tie) — dianggap ambigu.
4. **Validasi teks** (buang kosong/NaN).
5. **Split stratified 80% train / 10% val / 10% test** (`random_state=42`).

Hasil: **26.608 teks** (1.839 dibuang karena tie, 1 karena teks kosong), distribusi label
**~92% non-toxic vs ~8% toxic** — timpang, bawaan data aslinya. F1/precision/recall kelas *toxic*
lebih informatif daripada accuracy mentah untuk data se-timpang ini.

## Spesifikasi Hardware & OS

Berdasarkan eksperimen awal proyek ini, GPU yang terdeteksi adalah **NVIDIA GeForce RTX 3060**.
Konfigurasi default di `config.py` (`TRAIN_BATCH_SIZE=1`, `GRAD_ACCUM_STEPS=16`, gradient checkpointing
aktif) sengaja dibuat hemat VRAM untuk GPU kelas ini (~6–12 GB VRAM). **Jalankan `nvidia-smi` (atau cek
Task Manager di Windows) untuk memastikan spesifikasi mesinmu sendiri**, lalu sesuaikan bila perlu:

| Komponen | Rekomendasi minimum | Catatan |
|---|---|---|
| GPU | NVIDIA, ≥6 GB VRAM (CUDA) | Untuk `train.py`. Tanpa GPU, training tetap bisa jalan di CPU tapi **sangat lambat** (bisa berjam-jam) |
| RAM | ≥16 GB | Terutama untuk `shap_analysis.py` (memuat 3 model ONNX + proses SHAP) |
| Disk | ≥10 GB kosong | Model FP32 (~2 GB) + checkpoint Trainer + 3 model ONNX + dataset |
| OS | Linux / Windows / macOS | Diuji dengan pola Linux+CUDA; Windows/macOS bisa dipakai untuk CPU-only atau dengan penyesuaian driver CUDA masing-masing |
| Python | 3.10 – 3.12 | Cek dengan `python --version` |

Untuk melengkapi bagian ini dengan spesifikasi mesinmu sendiri, jalankan dan salin hasilnya ke sini:

**Linux / macOS**
```bash
python3 --version
nvidia-smi        # jika ada GPU NVIDIA (umumnya tidak tersedia di Mac)
```

**Windows** (Command Prompt atau PowerShell)
```powershell
python --version
nvidia-smi        # jika ada GPU NVIDIA
```

## Instalasi Environment

> **Catatan `python` vs `python3`**: di Linux/macOS, perintah `python` sering tidak tersedia
> (harus `python3`) — sedangkan di Windows umumnya `python` sudah benar. Setelah venv **aktif**,
> perbedaan ini tidak berlaku lagi: cukup pakai `python`/`pip` biasa di ketiga OS untuk semua
> perintah selanjutnya di README ini (`train.py`, `export_onnx.py`, dst).

### Linux

```bash
git clone <url-repo-ini>
cd qwen-indodiscourse-quant-xai

python3 -m venv venv
source venv/bin/activate

pip install --upgrade pip
pip uninstall -y torchvision    # jaga-jaga: proyek ini tidak butuh torchvision sama sekali;
                                 # torchvision versi tidak cocok adalah sumber umum error
                                 # "operator torchvision::nms does not exist" saat import transformers
pip install -r requirements.txt
```

### macOS

```bash
git clone <url-repo-ini>
cd qwen-indodiscourse-quant-xai

python3 -m venv venv
source venv/bin/activate

pip install --upgrade pip
pip uninstall -y torchvision
pip install -r requirements.txt
```

Catatan: Mac (terutama Apple Silicon/M1-M4) umumnya tidak punya GPU NVIDIA/CUDA — semua proses
otomatis fallback ke CPU dan **jauh lebih lambat**, khususnya `train.py` dan `shap_analysis.py`.

### Windows (PowerShell)

```powershell
git clone <url-repo-ini>
cd qwen-indodiscourse-quant-xai

python -m venv venv
venv\Scripts\Activate.ps1

pip install --upgrade pip
pip uninstall -y torchvision
pip install -r requirements.txt
```

Kalau PowerShell menolak menjalankan script (`... cannot be loaded because running scripts is
disabled on this system`), jalankan sekali ini di PowerShell **sebagai Administrator**, lalu ulangi
`venv\Scripts\Activate.ps1`:
```powershell
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
```

### Windows (Command Prompt / cmd.exe)

```bat
git clone <url-repo-ini>
cd qwen-indodiscourse-quant-xai

python -m venv venv
venv\Scripts\activate.bat

pip install --upgrade pip
pip uninstall -y torchvision
pip install -r requirements.txt
```

**Tanda venv sudah aktif** (semua OS): muncul `(venv)` di awal baris terminal.

**Sudah punya venv sendiri?** Aktifkan venv tersebut (pakai perintah aktivasi sesuai OS di atas)
lalu jalankan `pip install -r requirements.txt` di dalamnya — tidak perlu venv baru. Untuk memastikan
versi paket yang ter-install cocok dengan yang tertulis di `requirements.txt`, jalankan `pip list` dan
bandingkan; jika ada versi yang lebih lama dari batas minimum di `requirements.txt`, jalankan
`pip install --upgrade -r requirements.txt`.

Dependency kunci & alasannya ada di komentar `requirements.txt`. Dua yang sering terlewat:
- **`onnxscript`** — wajib untuk `torch.onnx.export(..., dynamo=True)` yang dipakai `export_onnx.py`.
- **`sympy`** — dipakai `onnxruntime.quantization.shape_inference.quant_pre_process` (persiapan PTQ Static).

## Alur Eksperimen

```
data/*.csv
    │  (src/data_utils.py: majority vote, drop tie, split 80/10/10)
    ▼
src/train.py  ───────────► artifacts/fp32/final/           (model + tokenizer PyTorch)
    │                      artifacts/evaluation/splits/     (train/val/test.csv)
    │                      artifacts/evaluation/fp32_pytorch_metrics.json
    ▼
src/export_onnx.py  ─────► artifacts/onnx/model_fp32.onnx
    │                      artifacts/evaluation/results_summary.json     (entry: FP32)
    │                      artifacts/evaluation/predictions_test.csv     (kolom: fp32_pred)
    ▼
src/quantize.py  ────────► artifacts/quantized/model_dynamic_int8.onnx
    │                      artifacts/quantized/model_static_int8.onnx
    │                      artifacts/evaluation/results_summary.json     (+ entry: Dynamic_INT8, Static_INT8)
    │                      artifacts/evaluation/predictions_test.csv     (+ kolom: dynamic_pred, static_pred)
    ▼
src/benchmark_latency.py ► artifacts/evaluation/latency_benchmark.json   (latensi & throughput FP32/Dynamic/Static)
    ▼
src/shap_analysis.py  ───► artifacts/shap/shap_token_level.pkl
    │                      artifacts/shap/shap_similarity_per_sample.csv, shap_similarity_summary.csv
    │                      artifacts/shap/plots/*.png, artifacts/shap/html/*.html
    ▼
notebooks/01_analisis.ipynb   <- HANYA membaca artifact di atas, tidak menghitung ulang apa pun
```

`results_summary.json` dan `predictions_test.csv` diisi **secara bertahap** (`export_onnx.py` menulis
entry `FP32`, `quantize.py` menambahkan `Dynamic_INT8` & `Static_INT8` tanpa menghapus entry yang sudah
ada) — jadi tiap script aman dijalankan ulang sendiri-sendiri kapan pun tanpa mengulang tahap lain.

## Menjalankan Script Terminal

Jalankan **berurutan** dari folder `src/` (atau `python src/nama_script.py` dari root repo):

```bash
cd src

python train.py              # [1] fine-tuning FP32 — PALING LAMA, butuh GPU idealnya
python export_onnx.py        # [2] export ke ONNX + sanity check PyTorch vs ONNX
python quantize.py           # [3] PTQ Dynamic & Static + evaluasi
python benchmark_latency.py  # [4] latensi & throughput (100 run x 2 batch size x 3 model, ~cepat)
python shap_analysis.py      # [5] interpretasi SHAP — lama, banyak forward pass per token
```

Tiap script mencetak progres ke terminal dan berhenti dengan pesan error yang jelas jika prasyaratnya
(artifact dari script sebelumnya) belum ada. Semua bisa dijalankan ulang kapan saja — hasil lama akan
ditimpa (atau digabung, khusus `results_summary.json`/`predictions_test.csv`).

## Menjalankan Notebook

Notebook ini **harus** dijalankan dengan kernel dari venv proyek ini (yang sudah ter-install
`requirements.txt`) — bukan kernel Python global/sistem, dan bukan venv proyek lain — karena butuh
`pandas`, `onnxruntime`, `transformers`, dst yang persis versinya.

### Opsi A — VSCode (direkomendasikan, sesuai setup kamu)

1. Buka folder repo ini di VSCode, buka `notebooks/01_analisis.ipynb`.
2. Klik **Select Kernel** di pojok kanan atas notebook → **Python Environments...** → pilih interpreter
   dari `venv/` proyek ini (biasanya muncul sebagai `venv (Python 3.x.x)` atau tertera path-nya, mis.
   `./venv/bin/python`).
3. Kalau `ipykernel` belum terdeteksi di venv itu, VSCode akan otomatis menawarkan **Install** — klik
   Install (ini akan menjalankan `pip install ipykernel` di venv yang sama, aman).
4. Jalankan **Run All**.

Kalau venv kamu tidak muncul di daftar, pastikan venv sudah diaktifkan minimal sekali di terminal
VSCode (`source venv/bin/activate` di Linux/macOS, atau `venv\Scripts\Activate.ps1` di Windows
PowerShell) atau arahkan manual lewat **Enter interpreter path...** ke file `venv/bin/python`
(Windows: `venv\Scripts\python.exe`).

### Opsi B — Jupyter Lab dari terminal

**Linux / macOS**
```bash
source venv/bin/activate
python -m ipykernel install --user --name qwen-indodiscourse --display-name "Python (qwen-indodiscourse)"
jupyter lab notebooks/01_analisis.ipynb
```

**Windows (PowerShell)**
```powershell
venv\Scripts\Activate.ps1
python -m ipykernel install --user --name qwen-indodiscourse --display-name "Python (qwen-indodiscourse)"
jupyter lab notebooks/01_analisis.ipynb
```

Setelah Jupyter Lab terbuka di browser: menu **Kernel → Change Kernel** → pilih
**"Python (qwen-indodiscourse)"**.

Jalankan semua sel dari atas ke bawah (*Run All*). Notebook otomatis mendeteksi lokasi folder `src/`
(baik dijalankan dengan working directory di root repo maupun di `notebooks/`) sehingga tidak perlu
mengatur `sys.path` manual. Notebook akan berhenti dengan pesan `assert` yang jelas jika artifact yang
dibutuhkan belum ada — jalankan script terminal terkait terlebih dahulu (lihat tabel di atas).

## Lokasi Artifact

| Artifact | Path | Dihasilkan oleh |
|---|---|---|
| Model + tokenizer FP32 (PyTorch) | `artifacts/fp32/final/` | `train.py` |
| Split data train/val/test | `artifacts/evaluation/splits/*.csv` | `train.py` |
| Metrik val/test FP32 (PyTorch) | `artifacts/evaluation/fp32_pytorch_metrics.json` | `train.py` |
| Model ONNX FP32 + tokenizer | `artifacts/onnx/` | `export_onnx.py` |
| Model ONNX Dynamic INT8 | `artifacts/quantized/model_dynamic_int8.onnx` | `quantize.py` |
| Model ONNX Static INT8 | `artifacts/quantized/model_static_int8.onnx` | `quantize.py` |
| Ukuran & metrik ketiga model | `artifacts/evaluation/results_summary.json` | `export_onnx.py` + `quantize.py` |
| Prediksi test set ketiga model | `artifacts/evaluation/predictions_test.csv` | `export_onnx.py` + `quantize.py` |
| Latensi & throughput ketiga model | `artifacts/evaluation/latency_benchmark.json` | `benchmark_latency.py` |
| Kemiripan SHAP per sampel & ringkasan | `artifacts/shap/shap_similarity_*.csv` | `shap_analysis.py` |
| Detail token-level SHAP | `artifacts/shap/shap_token_level.pkl` | `shap_analysis.py` |
| Plot & HTML SHAP (lampiran laporan) | `artifacts/shap/plots/`, `artifacts/shap/html/` | `shap_analysis.py` |

## Konfigurasi

Semua parameter (ukuran kalibrasi, `MAX_LENGTH`, jumlah epoch, jumlah sampel SHAP, dst) ada di
**`src/config.py`** — ubah di situ saja, otomatis konsisten dipakai oleh keempat script & notebook.
Tidak perlu mengubah path secara manual di file mana pun.

Satu yang berguna dari awal: **`DEV_SAMPLE_SIZE`** (default `None`) — set ke angka kecil (mis. `300`)
untuk uji coba seluruh pipeline dengan cepat sebelum commit ke run penuh yang lama. Lihat bagian
[Kustomisasi & FAQ](#kustomisasi--pertanyaan-umum-faq) di bawah untuk detail dan skenario lainnya.

## Kustomisasi & Pertanyaan Umum (FAQ)

### "Aku cuma mau tes semua script bisa jalan atau tidak, tanpa nunggu berjam-jam"

Set `DEV_SAMPLE_SIZE = 300` (atau angka kecil lain, minimal ~200) di `config.py`, lalu jalankan
pipeline seperti biasa (`train.py` → `export_onnx.py` → `quantize.py` → `benchmark_latency.py` →
`shap_analysis.py`). Seluruh
proses akan selesai dalam hitungan menit karena cuma pakai subset kecil data.

> ⚠️ **Hasilnya TIDAK valid untuk laporan/skripsi** — cuma untuk memastikan tidak ada error di kode.
> Setelah yakin semua lancar: **kembalikan `DEV_SAMPLE_SIZE = None`**, hapus folder `artifacts/`
> (atau minimal `artifacts/fp32/`, `artifacts/evaluation/`, `artifacts/onnx/`, `artifacts/quantized/`,
> `artifacts/shap/`), lalu jalankan ulang seluruh pipeline dari awal dengan data penuh.

### Ganti dataset

CSV harus punya kolom `text` plus kolom anotasi berformat list-per-annotator seperti `toxicity`
(mis. `"['0', '1']"` — satu nilai per annotator). Langkahnya:
1. Taruh CSV baru di `data/`, sesuaikan nama file di `RAW_CSV_PATH` (`config.py`).
2. Kalau nama kolom teks/anotasi berbeda, sesuaikan `ANNOTATION_COLS` dan `TARGET_COLUMN` di
   `config.py`/`data_utils.py`.
3. Kalau labelnya **sudah** satu nilai per baris (bukan list per annotator), lewati saja
   `majority_label`/`is_tie` di `data_utils.py` dan langsung pakai kolom label yang ada.
4. Kalau jumlah kelasnya lebih dari 2, lihat poin "Dataset multi-class" di bawah.

### Ganti target label (misal dari toxicity ke polarized)

Cukup ubah `TARGET_COLUMN = "polarized"` di `config.py` — tidak perlu ubah kode lain, karena
`data_utils.py` sudah generik terhadap nama kolom target (selama formatnya tetap list 0/1
per-annotator).

### Dataset/label bukan biner (multi-class)

Beberapa bagian kode saat ini mengasumsikan 2 kelas dan perlu disesuaikan manual:
- `data_utils.majority_label` / `is_tie` — logika *strict majority vote* untuk 0/1, perlu
  digeneralisasi (mis. pakai modus) untuk N kelas.
- `config.LABEL_NAMES` / `NUM_LABELS` — tinggal perbesar dict-nya, otomatis terpakai di training,
  evaluasi, dan notebook.
- `onnx_utils.evaluate` — ganti `average="binary"` jadi `average="macro"` atau `"weighted"`, dan
  hapus `pos_label=1`. Lakukan hal yang sama di `compute_metrics` pada `train.py`.
- `notebooks/01_analisis.ipynb` (bagian confusion matrix) — saat ini diasumsikan grid 2×2, perlu
  digeneralisasi mengikuti `len(LABEL_NAMES)`.

### Ganti model dasar (ukuran Qwen lain, atau arsitektur lain)

- **Ukuran Qwen2.5 lain** (mis. `Qwen/Qwen2.5-1.5B`): tinggal ganti `MODEL_NAME` di `config.py`.
  Perbesar `GRAD_ACCUM_STEPS` (atau kecilkan `MAX_LENGTH`) kalau VRAM tidak cukup untuk model yang
  lebih besar.
- **Arsitektur lain** (BERT, Llama, dst): `MODEL_NAME` bisa diganti bebas, tapi cek dua hal dulu:
  (1) apakah tokenizer-nya butuh `pad_token` diset manual seperti Qwen (BERT biasanya sudah punya),
  (2) wrapper `QwenForONNX` di `export_onnx.py` — mengasumsikan input `(input_ids, attention_mask)`
  dan output `.logits`, yang umumnya cocok untuk model klasifikasi HuggingFace lain juga, tapi tetap
  perlu dites ulang sanity check-nya.

### Menjalankan tanpa GPU (CPU-only), atau di Google Colab

Semua script otomatis fallback ke CPU kalau CUDA tidak tersedia (lebih lambat — pertimbangkan pakai
`DEV_SAMPLE_SIZE` dulu buat mastiin semuanya jalan). Untuk Google Colab: clone/upload repo ini ke
Colab, jalankan langkah instalasi yang sama di sel Colab (`!pip install -r requirements.txt`, dst),
lalu jalankan tiap script lewat `!python train.py`, dst — notebook `01_analisis.ipynb` tetap dipakai
di akhir untuk analisis saja. Simpan `artifacts/` ke Google Drive (`drive.mount("/content/drive")`,
lalu arahkan `ARTIFACTS_DIR` di `config.py` ke path Drive) supaya tidak hilang saat sesi Colab
berakhir.

### GPU AMD / Apple Silicon (MPS)

Script belum menangani backend selain CUDA secara eksplisit. Di Mac Apple Silicon, PyTorch mendukung
backend `mps`, tapi `Trainer` HuggingFace tidak selalu otomatis memakainya tergantung versi
`transformers`/`accelerate` — kalau mau memastikan GPU Apple Silicon terpakai, mungkin perlu setting
device manual. Tanpa penyesuaian ini, training akan fallback ke CPU (lambat, tapi tetap jalan benar).

### Training terputus di tengah jalan

`train.py` saat ini belum otomatis resume. Checkpoint per-epoch tetap tersimpan di
`artifacts/fp32/trainer_run/checkpoint-*`, jadi tidak hilang — untuk melanjutkan dari situ, edit
pemanggilan `trainer.train()` di `train.py` menjadi `trainer.train(resume_from_checkpoint=True)`.

### Sudah pernah run, mau eksperimen ulang dengan konfigurasi berbeda

Menjalankan ulang script mana pun **menimpa** artifact lama di folder terkait. Kalau mau membandingkan
beberapa eksperimen (mis. `MAX_LENGTH` berbeda), backup dulu foldernya sebelum run ulang:

```bash
# Linux/macOS
cp -r artifacts artifacts_eksperimen1
```
```powershell
# Windows PowerShell
Copy-Item -Recurse artifacts artifacts_eksperimen1
```

### `onnxruntime` vs `onnxruntime-gpu`

Jangan install keduanya sekaligus di venv yang sama (konflik). Untuk inference ONNX di GPU (opsional,
bukan wajib):
```bash
pip uninstall onnxruntime
pip install onnxruntime-gpu
```

### Reset satu tahap tertentu saja (tanpa reset semua)

Hapus folder/file artifact tahap terkait, script lain tidak perlu diulang:

| Mau reset tahap... | Hapus |
|---|---|
| Training | `artifacts/fp32/`, `artifacts/evaluation/splits/`, `artifacts/evaluation/fp32_pytorch_metrics.json` |
| Export ONNX + Kuantisasi | `artifacts/onnx/`, `artifacts/quantized/`, `artifacts/evaluation/results_summary.json`, `artifacts/evaluation/predictions_test.csv` |
| Latensi saja | `artifacts/evaluation/latency_benchmark.json` |
| SHAP saja | Isi folder `artifacts/shap/` |

## Troubleshooting

| Gejala | Penyebab umum | Solusi |
|---|---|---|
| `RuntimeError: operator torchvision::nms does not exist` / `Could not import module 'Trainer'` | `torchvision` ter-install dengan versi tidak cocok dengan `torch` | `pip uninstall -y torchvision` lalu `pip install -r requirements.txt` di venv yang sama, atau pakai venv baru |
| Export ONNX gagal / `dynamo=True` error | `onnxscript` belum ter-install, atau versi `torch` < 2.5 | `pip install onnxscript`; cek `python -c "import torch; print(torch.__version__)"` ≥ 2.5.0 |
| `CUDA out of memory` saat `train.py` | VRAM GPU tidak cukup | `TRAIN_BATCH_SIZE` sudah minimum (1); perbesar `GRAD_ACCUM_STEPS` di `config.py`, atau kecilkan `MAX_LENGTH` |
| `quant_pre_process` error / `ModuleNotFoundError: sympy` | `sympy` belum ter-install | `pip install sympy` |
| `shap_analysis.py` sangat lambat | SHAP partition explainer mahal secara komputasi | Kecilkan `N_SHAP_SAMPLES` dan/atau `SHAP_MAX_WORDS` di `config.py` |
| Angka latensi tidak stabil/jauh beda tiap run | Proses lain di CPU ikut membebani (browser, dll), atau `LATENCY_N_RUNS` kurang banyak | Tutup aplikasi berat lain saat benchmark; perbesar `LATENCY_N_RUNS` di `config.py`; baca `p50`/`p95` (tahan outlier), bukan cuma `mean` |
| ONNX Runtime print warning provider CUDA tidak ditemukan | Wajar bila tidak ada `onnxruntime-gpu` / GPU — otomatis fallback ke CPU | Abaikan, atau `pip install onnxruntime-gpu` untuk inference ONNX di GPU |
| Notebook: `FileNotFoundError: 'src' tidak ditemukan` | Notebook dibuka/dijalankan dari luar struktur repo | Pastikan working directory notebook ada di dalam repo ini (root atau `notebooks/`) |
| Notebook: `assert ... belum ada` | Artifact dari script terkait belum dibuat | Jalankan script yang disebut di pesan error (lihat tabel [Lokasi Artifact](#lokasi-artifact)) |
| `FileNotFoundError` dataset CSV | File CSV belum ada di `data/` | Pastikan `indotoxic2024_annotated_data_v2_final.csv` ada persis di `data/` (nama file harus sama) |

## Langkah Reproduksi Lengkap (Ringkas)

**Linux / macOS**
```bash
git clone <url-repo-ini> && cd qwen-indodiscourse-quant-xai
python3 -m venv venv && source venv/bin/activate
pip install --upgrade pip
pip uninstall -y torchvision
pip install -r requirements.txt

cd src
python train.py
python export_onnx.py
python quantize.py
python benchmark_latency.py
python shap_analysis.py
cd ..

jupyter lab notebooks/01_analisis.ipynb   # Run All
```

**Windows (PowerShell)**
```powershell
git clone <url-repo-ini>; cd qwen-indodiscourse-quant-xai
python -m venv venv; venv\Scripts\Activate.ps1
pip install --upgrade pip
pip uninstall -y torchvision
pip install -r requirements.txt

cd src
python train.py
python export_onnx.py
python quantize.py
python benchmark_latency.py
python shap_analysis.py
cd ..

jupyter lab notebooks/01_analisis.ipynb   # Run All
```

Setelah langkah terakhir, `notebooks/01_analisis.ipynb` berisi seluruh hasil: tabel & grafik
perbandingan ukuran/metrik FP32 vs Dynamic vs Static, serta analisis kemiripan interpretasi SHAP
antar ketiga model.

## Lisensi & Sitasi

### Lisensi kode

Repo ini belum menyertakan file `LICENSE` — tambahkan sesuai kebijakan kampus/pembimbingmu sebelum
dipublikasikan publik di GitHub. Untuk kode riset/skripsi, [MIT License](https://choosealicense.com/licenses/mit/)
adalah pilihan umum yang permisif (orang lain boleh pakai/modifikasi, wajib mencantumkan atribusi).
Cara pasang: buat file `LICENSE` di root repo, isi dengan teks lisensi pilihanmu (GitHub bisa
generate otomatis lewat tombol **Add file → Create new file → LICENSE** saat repo sudah di-push).

### Sitasi dataset

Dataset yang dipakai repo ini (IndoToxic2024) berasal dari:

```bibtex
@article{susanto2024indotoxic,
  title   = {IndoToxic2024: A Demographically-Enriched Dataset of Hate Speech and Toxicity
             Types for Indonesian Language},
  author  = {Susanto, Lucky and Wijanarko, Musa Izzanardi and Pratama, Prasetia Anugrah
             and Hong, Traci and Idris, Ika and Aji, Alham Fikri and Wijaya, Derry},
  journal = {arXiv preprint arXiv:2406.19349},
  year    = {2024}
}
```

Cantumkan sitasi ini di bab metodologi/dataset skripsimu. Cek juga lisensi asli dataset (lihat
halaman [Hugging Face](https://huggingface.co/datasets/Exqrch/IndoDiscourse) atau paper-nya) sebelum
menyertakan file CSV mentah ke repo publik — beberapa dataset riset punya syarat penggunaan (mis.
non-komersial, wajib sitasi) yang perlu dipatuhi.

### Mengutip skripsi/repo ini

Kalau repo ini nanti dikutip orang lain, sediakan info sitasi di sini setelah judul & identitas
skripsimu final, misalnya:

```bibtex
@misc{<nama-belakang><tahun>,
  title  = {Analisis Pengaruh Kuantisasi terhadap Explainable AI (XAI) pada Model Qwen2.5-0.5B
            untuk Klasifikasi Toksisitas Berbahasa Indonesia},
  author = {<Nama Lengkap>},
  school = {<Nama Institusi>},
  year   = {<Tahun>},
  note   = {\\url{<url-repo-ini>}}
}
```
