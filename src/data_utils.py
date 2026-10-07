"""Load & preprocessing dataset IndoToxic2024/IndoDiscourse.

Logika agregasi label (majority vote + drop tie) mengikuti persis metodologi pada
`1.ipynb`: setiap teks dianotasi beberapa annotator (list nilai 0/1 per kolom).
Label final = 1 jika mayoritas annotator memberi 1 (strict majority); teks dengan
suara seri (tie, mis. 1-1 dari 2 annotator) dibuang karena dianggap ambigu.
"""
import ast

import pandas as pd
from sklearn.model_selection import train_test_split

import config as cfg

ANNOTATION_COLS = [
    "annotators_id",
    "is_noise_or_spam_text",
    "related_to_election_2024",
    "toxicity",
    "polarized",
    "profanity_obscenity",
    "threat_incitement_to_violence",
    "insults",
    "identity_attack",
    "sexually_explicit",
]


def load_raw(csv_path=None) -> pd.DataFrame:
    csv_path = csv_path or cfg.RAW_CSV_PATH
    if not csv_path.exists():
        raise FileNotFoundError(
            f"Dataset tidak ditemukan di: {csv_path}\n"
            "Letakkan 'indotoxic2024_annotated_data_v2_final.csv' di folder data/ (lihat README.md)."
        )
    return pd.read_csv(csv_path)


def parse_annotations(df: pd.DataFrame) -> pd.DataFrame:
    """Ubah kolom string list ("['0', '1']") menjadi list Python asli."""
    return df[ANNOTATION_COLS].map(ast.literal_eval)


def majority_label(values) -> int:
    """Strict majority vote: 1 jika jumlah anotasi '1' > separuh total anotasi."""
    ints = list(map(int, values))
    return int(sum(ints) > len(ints) / 2)


def is_tie(values) -> bool:
    """True jika suara persis seri (mis. 1-1 dari 2 annotator)."""
    ints = list(map(int, values))
    return sum(ints) * 2 == len(ints)


def build_labeled_dataset(df: pd.DataFrame = None) -> pd.DataFrame:
    """Pipeline lengkap: parse anotasi -> agregasi -> drop tie -> validasi teks.

    Return DataFrame dengan kolom: text_id, text, label, label_name.
    """
    df = df.copy() if df is not None else load_raw()
    parsed = parse_annotations(df)

    target = cfg.TARGET_COLUMN
    df["label"] = parsed[target].apply(majority_label)
    tie_mask = parsed[target].apply(is_tie)

    n_before = len(df)
    df = df.loc[~tie_mask].copy()
    n_tie = int(tie_mask.sum())

    df = df.dropna(subset=["text"])
    df = df[df["text"].str.strip() != ""].copy()

    df["label_name"] = df["label"].map(cfg.LABEL_NAMES)
    df = df[["text_id", "text", "label", "label_name"]].reset_index(drop=True)

    print(f"Data awal        : {n_before}")
    print(f"Dibuang (tie)    : {n_tie}")
    print(f"Dibuang (teks NA): {n_before - n_tie - len(df)}")
    print(f"Data final       : {len(df)}")

    # Mode cepat (opsional) untuk uji coba SELURUH pipeline (train -> export -> quantize -> shap)
    # dalam hitungan menit, bukan jam -- berguna buat mastiin tidak ada error dulu sebelum run
    # penuh yang lama. Aktifkan dengan set DEV_SAMPLE_SIZE di config.py (default None = nonaktif).
    if getattr(cfg, "DEV_SAMPLE_SIZE", None) is not None and cfg.DEV_SAMPLE_SIZE < len(df):
        df, _ = train_test_split(
            df, train_size=cfg.DEV_SAMPLE_SIZE, stratify=df["label"], random_state=cfg.RANDOM_SEED
        )
        df = df.reset_index(drop=True)
        print(f"[DEV_SAMPLE_SIZE={cfg.DEV_SAMPLE_SIZE} aktif] Memakai subset kecil untuk uji cepat "
              f"pipeline -- JANGAN dipakai untuk hasil final skripsi. Set None di config.py untuk "
              f"memakai seluruh data.")

    return df


def stratified_split(df: pd.DataFrame):
    """80% train, 10% val, 10% test — stratified terhadap label (seed = cfg.RANDOM_SEED)."""
    train_df, temp_df = train_test_split(
        df, test_size=cfg.TEST_SIZE, random_state=cfg.RANDOM_SEED, stratify=df["label"]
    )
    val_df, test_df = train_test_split(
        temp_df, test_size=cfg.VAL_TEST_SPLIT, random_state=cfg.RANDOM_SEED, stratify=temp_df["label"]
    )
    return (
        train_df.reset_index(drop=True),
        val_df.reset_index(drop=True),
        test_df.reset_index(drop=True),
    )


def load_and_preprocess():
    """Shortcut: load CSV mentah -> dataset berlabel -> split train/val/test."""
    df = build_labeled_dataset()
    return stratified_split(df)
