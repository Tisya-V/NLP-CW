import pandas as pd
import os
import constants
import numpy as np

def load_full_dataset():
    """ Load the full dataset from the official PCL .tsv file
        From: https://github.com/CRLala/NLPLabs-2024/blob/main/Dont_Patronize_Me_Trainingset/dontpatronizeme_pcl.tsv
        Returns a pandas DataFrame with a binary_label column (0 for non-patronizing, 1 for patronizing)
        Where the original label is >= 2, we consider it patronizing (label=1), otherwise non-patronizing (label=0)
        (As in the original paper)
    """
    pcl_cols = ["par_id", "art_id", "keyword", "country_code", "text", "label"]
    df_full = pd.read_csv(
        os.path.join(constants.DATA_DIR, "dontpatronizeme_pcl.tsv"),
        sep="\t",
        names=pcl_cols,
        skiprows=4,   # skip the header/comment rows
        dtype={"par_id": str}
    )
    df_full["binary_label"] = (df_full["label"] >= 2).astype(int)
    # df_full.head()
    return df_full

def load_train_dev_test_splits():
    """ Load the train/dev/test splits based on the par_id lists provided in the official CSV files
        From: 
        - https://github.com/Perez-AlmendrosC/dontpatronizeme/blob/master/semeval-2022/practice%20splits/train_semeval_parids-labels.csv
        - https://github.com/Perez-AlmendrosC/dontpatronizeme/blob/master/semeval-2022/practice%20splits/dev_semeval_parids-labels.csv
        - https://github.com/Perez-AlmendrosC/dontpatronizeme/blob/master/semeval-2022/TEST/task4_test.tsv
        Returns tuple of DataFrames: (train_df, dev_df, test_df)
    """
    df_full = load_full_dataset()
    train_ids = pd.read_csv(
        os.path.join(constants.DATA_DIR, "train_semeval_parids-labels.csv"),
        dtype={"par_id": str}
    )
    dev_ids = pd.read_csv(
        os.path.join(constants.DATA_DIR, "dev_semeval_parids-labels.csv"),
        dtype={"par_id": str}
    )
    train_df = df_full[df_full["par_id"].isin(train_ids["par_id"])].reset_index(drop=True)
    dev_df   = df_full[df_full["par_id"].isin(dev_ids["par_id"])].reset_index(drop=True)

    test_cols = ["par_id", "art_id", "keyword", "country_code", "text"]
    test_df = pd.read_csv(
        os.path.join(constants.DATA_DIR, "task4_test.tsv"),
        sep="\t",
        names=test_cols,
        skiprows=0,
        dtype={"par_id": str}
    )
    print("Data loaded successfully")
    print(f"Train: {len(train_df)} | Dev: {len(dev_df)} | Test: {len(test_df)}")
    return train_df, dev_df, test_df

def clean_df(df, split_name, clean_short_words: bool = False, short_text_threshhold: int = 3):
    df = df.copy()

    # Force non-string types to NaN so str methods work reliably
    df["text"] = df["text"].str.strip().str.replace(r"\s+", " ", regex=True).replace("", pd.NA)
    original_len = len(df)
    df["_original_index"] = np.arange(len(df))  # help track dropped cols
    null_mask  = df["text"].isna()
    empty_mask = df["text"].str.strip().eq("")
    short_mask = df["text"].str.split().str.len() < short_text_threshhold
    drop_cols = [c for c in ["par_id", "text"] if c in df.columns]

    print(f"{split_name}: {null_mask.sum()} NaN | {empty_mask.sum()} empty | "
          f"{short_mask.sum()} <{short_text_threshhold} words")

    if short_mask.any():
        if "binary_label" in df.columns:
            print(f"Short rows (<{short_text_threshhold} words):", df[short_mask][["par_id", "text", "binary_label"]].to_string())
        else:
            print(f"Short rows: (<{short_text_threshhold} words)", df[short_mask][["par_id", "text"]].to_string())

        if clean_short_words:
            print(f"  Dropping {short_mask.sum()} rows:")
            print(df[short_mask][drop_cols].to_string())
            df = df[~short_mask].reset_index(drop=True)

    bad_mask = null_mask | empty_mask
    if bad_mask.any():
        print(f"  Dropping {bad_mask.sum()} rows:")
        print(df[bad_mask][drop_cols].to_string())
        df = df[~bad_mask].reset_index(drop=True)

    kept_indices = df["_original_index"].tolist()
    df = df.drop(columns=["_original_index"]).reset_index(drop=True)
    
    return df, kept_indices, original_len

def load_and_clean_data():
    train_df, dev_df, test_df = load_train_dev_test_splits()
    train_df, _, _ = clean_df(train_df, "train")
    dev_df, dev_kept, dev_oglen   = clean_df(dev_df,   "dev") 
    test_df, test_kept, test_oglen  = clean_df(test_df,  "test")
    return train_df, dev_df, dev_kept, dev_oglen, test_df, test_kept, test_oglen

if __name__ == "__main__":
    load_and_clean_data()