# CNN + LSTM Embryo Phase Model (Staged Training)

This README documents the **sequence model** implemented in **`lstm-on-embryo-dataset.ipynb`**.

The goal is to classify embryo development into **16 ordered phases** using **short frame sequences**:
- A **CNN encoder** (MobileNetV2 feature extractor)
- An **LSTM** over frame features
- A **classification head** that predicts the phase label

It is designed to be practical on local hardware by using **staged training** (≈8% of videos per stage) and an optional **step cap** (≈3000 steps/epoch).

---

## 1) Dataset layout (expected)

Your workspace is expected to contain these folders:

- `embryo_dataset/` — video folders containing frames
  - Example: `embryo_dataset/AA83-7/AA83-7_RUN123.jpg`
  - The notebook expects frame filenames to contain a run index matching this pattern:
    - `*_RUN<number>.jpg` (or `.jpeg`, `.png`)

- `embryo_dataset_annotations/` — per-video phase CSV files
  - One CSV per video: `embryo_dataset_annotations/<VIDEO_ID>_phases.csv`
  - Format: **no header**, each row is:

    ```text
    PHASE_NAME, START_RUN, END_RUN
    ```

  - Example row:

    ```text
    t4, 120, 180
    ```

The 16 phase names must match the notebook’s `PHASES` list.

---

## 2) Model summary

### Input
- A sample is a **sequence of frames** of length `SEQ_LEN` (default: 5)
- Each training sample corresponds to a specific annotated run index (frame time) inside a video

### Architecture
- CNN: `torchvision.models.mobilenet_v2(weights='DEFAULT')`
  - The notebook freezes CNN parameters to reduce VRAM
  - The classification head is removed and replaced with `nn.Identity()` to use it as a feature extractor

- LSTM: consumes the sequence of CNN features
- Head: MLP + linear to 16 logits

---

## 3) Loss function (Chronological / ordinal-aware)

The notebook uses a **hybrid loss**:

- Component A: **Cross Entropy** (standard classification)
- Component B: **Ordinal distance loss** based on **CDF SmoothL1**
  - Computes a probability distribution over ordered phases
  - Converts it to a cumulative distribution (CDF)
  - Penalizes differences between predicted CDF and target CDF via SmoothL1

### Beta warmup + ramp
The ordinal term is scaled by `beta(epoch)`:
- Warmup: epoch 0 → `beta = 0`
- Ramp: next 2 epochs → `beta` linearly increases to `BETA_MAX` (default: 0.30)

This prevents the ordinal term from destabilizing very early training.

---

## 4) Training design (the key part)

Training is done in **sequential stages** so your machine can finish progressively and resume safely.

### 4.1 Fixed 70/15/15 video split
The split is performed at the **video level**:
- `TRAIN_SPLIT = 0.70`
- `VAL_SPLIT = 0.15`
- `TEST_SPLIT = 0.15`

This avoids leakage (frames from the same embryo video never appear in both train and val/test).

### 4.2 8% stage definition (video-level)
The training set is broken into multiple stages.

- `STAGE_TOTAL_VIDEO_FRACTION = 0.08`
- Stage size is computed as:
  - `stage_size_vids = round(0.08 * total_number_of_videos)`

Stages are created by shuffling the **train video list** (using `SPLIT_SEED`) and taking chunks of `stage_size_vids`.

Practical note:
- The **last stage can be smaller** if the train set size doesn’t divide evenly.

### 4.3 ~3000 steps/epoch cap (sample-level)
Even with 8% of videos per stage, some videos may contain many annotated frames.

To keep each epoch short, the notebook can cap work per epoch using:
- `LIMIT_TRAIN_STEPS = True`
- `TARGET_TRAIN_STEPS = 3000`

How it works:
- It builds the *full* training dataset for the stage
- It then selects a **fixed random subset of samples** of size:

  ```text
  subset_len = min(dataset_len, TARGET_TRAIN_STEPS * BATCH_SIZE)
  ```

- This subset is saved to disk, so the same stage uses the same subset across restarts

This makes per-epoch runtime predictable.

---

## 5) Outputs (what gets saved + how to read it)

All LSTM run artifacts are saved under:

- `outputs_lstm/`

### 5.1 Split files (reproducibility)
Saved in:
- `outputs_lstm/splits/`

Files:
- `train_vids.txt`, `val_vids.txt`, `test_vids.txt`
- `train_stage_01_vids.txt`, `train_stage_02_vids.txt`, …

These are the exact video IDs used per split and per stage.

### 5.2 Subset indices (to enforce ~3000 steps)
Saved in:
- `outputs_lstm/subsets/`

Files:
- `train_subset_indices_stageXX_nYYYY_seedZZ.npy`
- `train_subset_meta_stageXX_nYYYY_seedZZ.json`

Meaning:
- `nYYYY` is the number of samples used in that stage per epoch
- The `.json` includes the dataset length and the target steps used

### 5.3 Per-stage resume checkpoints (model + optimizer + scheduler + history)
Saved in:
- `outputs_lstm/latest_checkpoint_stageXX.pth`

Each file contains:
- `model_state_dict`
- `optimizer_state_dict`
- `scheduler_state_dict`
- `history` (loss curves across all stages so far)
- `global_best_val` and `best_model_path`
- `stage_epoch_next` / `global_epoch_next` for resuming

These are the files you use to resume training after an interruption.

### 5.4 Global best weights (weights-only)
Saved in:
- `outputs_lstm/best_cnn_lstm.pth`

Meaning:
- This is the **single best model** found across all stages and epochs, based on **lowest validation total loss**.
- It does *not* include optimizer/scheduler state.

### 5.5 Final metrics file (for submission)
Saved in:
- `outputs_lstm/final_metrics.json`

This file is generated by the final evaluation cell and includes:
- `test.exact_acc`
- `test.tol_acc_pm1`
- `test.total_loss`
- metadata fields indicating whether evaluation was approximate (`is_approx_eval`) and how many test samples were used.

### 5.6 Console output (what to expect)
During training you should see logs like:

- Video split sizes: `Train/Val/Test`
- Stage summary: stage size and number of stages
- For each stage:
  - dataset sample counts (`full=... used=...`)
  - `~steps/epoch` (should be ≈`TARGET_TRAIN_STEPS` if the dataset is large enough)
  - per-epoch summary:

    ```text
    Train total (ce, ord) + tol(±1)
    Val total (ce, ord) + tol(±1)
    ```

Tolerance accuracy (±1) is useful for ordinal labels, because predicting an adjacent phase is less severe than predicting a distant phase.

### 5.7 Plots (graphs)
The plotting cell in `lstm-on-embryo-dataset.ipynb` is designed to work even after a restart:
- If `history` is still in memory, it plots directly
- Otherwise it loads `history` from the **latest** `outputs_lstm/latest_checkpoint_stage*.pth`

It plots:
- Total loss + beta schedule
- Cross Entropy component
- Ordinal (CDF SmoothL1) component

#### How to interpret the curves (what “good” looks like)

**Panel 1: Total Loss + Beta Schedule**
- **Validation loss trending down and staying stable** is the main indicator of generalization. If training loss keeps dropping but validation loss rises steadily, that’s classic overfitting.
- A **stable or slowly improving validation loss** over many epochs is a sign your training pipeline is stable (no exploding gradients / divergence).
- The **beta (dashed line)** should follow the intended shape:
  - Start near 0 during warmup (so CE dominates early)
  - Ramp up to `BETA_MAX` over the configured ramp window
  - Then remain steady

**Why training loss can look “spiky” (sawtooth pattern)**
- A repeating drop → spike → drop pattern in **training loss** is often normal in this notebook because training is not perfectly stationary:
  - Each stage trains on a different **subset of videos** (8% chunks of the train split).
  - If `LIMIT_TRAIN_STEPS=True`, each stage also uses a **fixed random subset of samples** (which changes per stage).
- When the model sees a “harder” stage/subset, training loss can jump, then fall again as it adapts. This is usually a healthy sign of continued learning rather than instability.

**Panels 2–3: CE vs Ordinal loss**
- It’s expected that **CE** and **ordinal** components have different absolute scales.
- What matters is that both are **well-behaved** (no runaway growth) and that increasing beta does not cause sudden divergence.
- If ordinal loss collapses to ~0 immediately while CE stays high, or ordinal loss explodes when beta ramps up, that suggests imbalance or learning-rate issues.

**The “plateau” (when validation stops improving)**
- If validation loss flattens for many epochs, it usually means you’ve reached the limit of what the current setup can extract.
- In this notebook, the CNN encoder is **frozen**, so a plateau can indicate:
  - The LSTM/head have learned most of what they can from the fixed CNN features
  - Additional epochs mostly refine training loss without improving validation

Practical takeaway:
- Once validation has plateaued for a long stretch (and `ReduceLROnPlateau` has already reduced LR), it’s typically better to stop and keep the **best saved weights** (`outputs_lstm/best_cnn_lstm.pth`) rather than training indefinitely.

---


## 6) Latest fast submission result

Using the final evaluation cell in fast mode:
- `FAST_SUBMISSION_MODE=True`
- `MAX_TEST_SAMPLES=4000`
- `RUN_VAL_EVAL=False`
- `EVAL_BATCH_SIZE=64`

Observed run:
- Test samples used: `4000 / 44709`
- Final TEST total loss: `1.8382`
- Final TEST CE loss: `1.8285`
- Final TEST ordinal loss: `0.0325`
- Final TEST Exact Accuracy: `43.55%`
- Final TEST Tolerance Accuracy (±1): `66.88%`

Saved output file:
- `outputs_lstm/final_metrics.json`

Note:
- This is a **fast approximate** submission metric (deterministic capped subset).

