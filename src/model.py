# %%
import torch
import numpy as np
import pandas as pd
from torch import nn
import torch
print("cuda available:", torch.cuda.is_available())
print("imported torch")
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer,
)
print("imported tokeniser")
from sklearn.metrics import f1_score, classification_report
from sklearn.utils import resample
from sklearn.utils.class_weight import compute_class_weight
from sklearn.model_selection import train_test_split
from datasets import Dataset
print("cuda" if torch.cuda.is_available() else "cpu")

# !git clone https://github.com/Tisya-V/NLP-CW.git
# import sys
# sys.path.insert(0, "/content/NLP-CW/src")

import utils

# !nvidia-smi

# %%
RANDOM_SEED = 47

train_df, dev_df, test_df = utils.load_and_clean_data()

train_df, val_df = train_test_split(
    train_df,
    test_size=0.15,
    stratify=train_df["binary_label"],  # ensures same class ratio in both splits
    random_state=RANDOM_SEED
)
# !pip install nlpaug nltk
import nlpaug.augmenter.word as naw
import nltk

# Synonym replacement: replace up to 15% of words, max 3 swaps per sentence
sr_aug = naw.SynonymAug(
    aug_src='wordnet',
    aug_p=0.15,
    aug_max=3
)

def augment_pcl_rows(df, augmenter, n_copies=1, seed=RANDOM_SEED):
    pcl_rows = df[df["binary_label"] == 1].copy()
    augmented = []
    for _, row in pcl_rows.iterrows():
        for _ in range(n_copies):
            try:
                new_text = augmenter.augment(row["text"])[0]
            except Exception:
                new_text = row["text"]  # fallback: keep original if augmentation fails
            augmented.append({"text": new_text, "binary_label": 1})
    return pd.DataFrame(augmented)

aug_df = augment_pcl_rows(train_df, sr_aug, n_copies=3)
train_df_aug = pd.concat([train_df, aug_df], ignore_index=True).sample(
    frac=1, random_state=RANDOM_SEED
).reset_index(drop=True)

print(f"Original: {train_df['binary_label'].value_counts().to_dict()}")
print(f"Augmented: {train_df_aug['binary_label'].value_counts().to_dict()}")

# train_df = train_df_aug

# majority = train_df[train_df["binary_label"] == 0]
# minority = train_df[train_df["binary_label"] == 1]

# # Upsample minority to match majority
# minority_upsampled = resample(
#     minority,
#     replace=True,                    # sample with replacement
#     n_samples=len(majority) // 2,         # match majority class size
#     random_state=RANDOM_SEED
# )

# train_balanced = pd.concat([majority, minority_upsampled]).sample(
#     frac=1, random_state=RANDOM_SEED
# ).reset_index(drop=True)

# print(f"Original: {len(train_df)} | Balanced: {len(train_balanced)}")
# print(train_balanced["binary_label"].value_counts())

# %%
MODEL_NAME = "roberta-base"
tokenizer  = AutoTokenizer.from_pretrained(MODEL_NAME, use_fast=False)

def tokenize(df):
    enc = tokenizer(
        df["text"].tolist(),
        padding="max_length",
        truncation=True,
        max_length=256,
    )
    if "binary_label" in df.columns:
        enc["labels"] = df["binary_label"].tolist()
    return Dataset.from_dict(enc)

train_dataset = tokenize(train_df_aug)
val_dataset   = tokenize(val_df)
dev_dataset   = tokenize(dev_df)

# %%
labels_array = train_df["binary_label"].values
class_weights = compute_class_weight(
    class_weight="balanced",
    classes=np.array([0, 1]),
    y=labels_array
)
print(f"Class weights → 0: {class_weights[0]:.3f}, 1: {class_weights[1]:.3f}")

# Move to device for use in loss
device = "cuda" if torch.cuda.is_available() else "cpu"
print("Device:", device)
device = torch.device(device)
class_weights_tensor = torch.tensor(class_weights, dtype=torch.float).to(device)

# %%
class WeightedTrainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        labels  = inputs.get("labels")
        outputs = model(**inputs)
        logits  = outputs.get("logits")

        loss_fct = nn.CrossEntropyLoss(weight=class_weights_tensor.to(logits.dtype))
        loss = loss_fct(logits.view(-1, self.model.config.num_labels), labels.view(-1))

        return (loss, outputs) if return_outputs else loss

# %%

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=-1)
    # F1 of positive class only — matches the task metric
    f1 = f1_score(labels, preds, pos_label=1)
    return {"f1_pcl": f1}



# %%
from itertools import product
from scipy.special import softmax


# ── Grid definition ──────────────────────────────────────────────────────────
param_grid = {
    "learning_rate"    : [2e-5, 1e-5, 5e-6],
    "grad_acc_steps"   : [1, 2, 4],
    "num_train_epochs" : [3, 4, 5],
}

keys   = list(param_grid.keys())
combos = list(product(*param_grid.values()))
print(f"Total combinations: {len(combos)}")

# ── Results store ────────────────────────────────────────────────────────────
results = []

for i, combo in enumerate(combos):
    params = dict(zip(keys, combo))
    print(f"\n▶ Running {i} / {len(combos)}: {params}")

    model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME, num_labels=2)

    training_args = TrainingArguments(
        output_dir                  = "./model",
        num_train_epochs            = params["num_train_epochs"],
        per_device_train_batch_size = 8,
        gradient_accumulation_steps = params["grad_acc_steps"],
        per_device_eval_batch_size  = 32,
        learning_rate               = params["learning_rate"],
        warmup_ratio                = 0.1,
        weight_decay                = 0.01,
        eval_strategy               = "epoch",
        save_strategy               = "no",          # don't save every sweep run
        load_best_model_at_end      = False,         # manual best tracking below
        bf16                        = False,
        fp16                        = False,
    )

    trainer = WeightedTrainer(
        model           = model,
        args            = training_args,
        train_dataset   = train_dataset,
        eval_dataset    = val_dataset,
        compute_metrics = compute_metrics,
    )

    trainer.train()

    # ── Threshold tuning on val for this config ───────────────────────────
    val_out      = trainer.predict(val_dataset)
    val_probs    = softmax(val_out.predictions, axis=-1)[:, 1]
    val_labels   = val_df["binary_label"].values

    best_t, best_f1 = 0.5, 0.0
    for t in np.linspace(0.05, 0.95, 91):
        preds = (val_probs >= t).astype(int)
        f1    = f1_score(val_labels, preds, pos_label=1)
        if f1 > best_f1:
            best_f1, best_t = f1, t

    results.append({**params, "val_f1": best_f1, "best_thresh": best_t})
    print(f"   Val F1: {best_f1:.4f}  |  Threshold: {best_t:.2f}")

    # Free GPU memory between runs
    del model, trainer
    torch.cuda.empty_cache()

# ── Summary ───────────────────────────────────────────────────────────────────
results_df = pd.DataFrame(results).sort_values("val_f1", ascending=False)
print("\n── Sweep Results ──")
print(results_df.to_string(index=False))


# %%
# Retrain best model
best = results_df.iloc[0]
print(f"\nBest config: LR={best.learning_rate}, epochs={best.num_train_epochs}, "
      f"grad_acc_steps={best.grad_acc_steps}, F1={best.val_f1:.4f}, thresh={best.best_thresh:.2f}")

# Retrain on full train_df_aug with best params + load best epoch checkpoint
model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME, num_labels=2)
training_args = TrainingArguments(
    output_dir              = "./model_best",
    num_train_epochs        = int(best.num_train_epochs),
    per_device_train_batch_size = 8,
    gradient_accumulation_steps=best.grad_acc_steps,
    per_device_eval_batch_size  = 32,
    learning_rate           = best.learning_rate,
    warmup_ratio            = 0.1,
    weight_decay            = 0.01,
    eval_strategy           = "epoch",
    save_strategy           = "epoch",
    load_best_model_at_end  = True,
    metric_for_best_model   = "f1_pcl",
    bf16=False, fp16=False,
    logging_steps           = 1,
    report_to               = "none",
)
trainer = WeightedTrainer(
    model=model, 
    args=training_args,
    train_dataset=train_dataset, 
    eval_dataset=val_dataset,
    compute_metrics=compute_metrics,
)

trainer.train()


# %%
# from scipy.special import softmax

# # threshhold tuning on val_df
# val_output   = trainer.predict(val_dataset)
# val_logits   = val_output.predictions
# val_probs    = softmax(val_logits, axis=-1)[:, 1]   # P(PCL)
# val_labels   = val_df["binary_label"].values

# best_thresh, best_f1 = 0.5, 0.0
# for t in np.linspace(0.05, 0.95, 91):
#     preds = (val_probs >= t).astype(int)
#     f1    = f1_score(val_labels, preds, pos_label=1)
#     if f1 > best_f1:
#         best_f1, best_thresh = f1, t

# print(f"Best threshold: {best_thresh:.2f}  |  Val F1: {best_f1:.4f}")

def get_predictions(trainer, dataset, threshold = 0.5):
    preds_output = trainer.predict(dataset)
    logits = preds_output.predictions
    probs = softmax(logits, axis=1)[:, 1]
    preds = (probs >= threshold).astype(int)
    return preds

# Dev predictions
dev_preds = get_predictions(trainer, dev_dataset, threshold = best.best_thresh)
print(classification_report(dev_df["binary_label"].values, dev_preds, target_names=["No PCL", "PCL"]))
np.savetxt("dev.txt", dev_preds.astype(int), fmt="%d")

# Test predictions (no labels)
test_dataset = tokenize(test_df)
test_preds = get_predictions(trainer, test_dataset, threshold = best.best_thresh)
np.savetxt("test.txt", test_preds.astype(int), fmt="%d")



