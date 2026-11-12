# 4SRec

## Repository Layout

```text
4SRec/
├── requirements.txt
├── stage1_FT_ServiceLLM/
│   ├── data/
│   ├── model/
│   ├── prepare_data.py
│   ├── sample_preferences.py
│   ├── main.py                 # Fine-tune ServiceLLM
│   ├── extract_embeddings.py
│   └── reduce_embeddings.py
└── stage2_Downstream_Recommendation/
    ├── data/                   # Includes the processed Scientific dataset
    ├── model/
    └── main.py                 # Train and evaluate LLGCN and InteRec
```

### Step 1. Install Dependencies

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt

STAGE1=stage1_FT_ServiceLLM
STAGE2=stage2_Downstream_Recommendation
```

## Stage 1. Fine-Tune ServiceLLM

### Step 2. Download and Preprocess a Dataset

**Input:**  A ratings-only CSV and product metadata gzip for an Amazon category, or the review and business JSON files for Yelp. Place both files under `stage1_FT_ServiceLLM/data/raw/`.

| Dataset | `DATASET` | Raw files |
| --- | --- | --- |
| Amazon Industrial and Scientific | `Scientific` | [Amazon download page](https://cseweb.ucsd.edu/~jmcauley/datasets/amazon_v2/) |
| Amazon Arts, Crafts and Sewing | `Arts` | [Amazon download page](https://cseweb.ucsd.edu/~jmcauley/datasets/amazon_v2/) |
| Amazon Office Products | `Office` | [Amazon download page](https://cseweb.ucsd.edu/~jmcauley/datasets/amazon_v2/) |
| Yelp Open Dataset | `Yelp` | [Yelp download page](https://www.yelp.com/dataset/documentation/main) |

```bash
mkdir -p "$STAGE1/data/raw"
```

After downloading the files, choose one dataset configuration.

```bash
# Amazon Scientific
DATASET=Scientific
CATEGORY=Industrial_and_Scientific

# To use Arts or Office, uncomment the corresponding line.
# DATASET=Arts;   CATEGORY=Arts_Crafts_and_Sewing
# DATASET=Office; CATEGORY=Office_Products

INTERACTIONS="$STAGE1/data/raw/${CATEGORY}.csv"
METADATA="$STAGE1/data/raw/meta_${CATEGORY}.json.gz"
```

<details>
<summary>Yelp configuration</summary>

```bash
DATASET=yelp
INTERACTIONS="$STAGE1/data/raw/yelp_academic_dataset_review.json"
METADATA="$STAGE1/data/raw/yelp_academic_dataset_business.json"
```

</details>

```bash
SLUG=$(printf '%s' "$DATASET" | tr '[:upper:]' '[:lower:]')
PROCESSED="$STAGE1/data/processed"
REC_DATA="$STAGE2/data/rebuilt"

python "$STAGE1/prepare_data.py" --dataset "$DATASET" \
  --interactions "$INTERACTIONS" --metadata "$METADATA" \
  --data-root "$PROCESSED"
```

**Output:**  The interaction splits and ordered metadata for the selected dataset.

```text
stage1_FT_ServiceLLM/data/processed/<dataset>/
├── train.pkl     # User IDs mapped to training item IDs
├── test.pkl      # User IDs mapped to held-out item IDs
├── users.jsonl   # User records in numeric ID order
└── items.jsonl   # Item metadata in numeric ID order
```

### Step 3. Sample Preference Triplets

**Input:**  `train.pkl`, `users.jsonl`, and `items.jsonl` from Step 2.

```bash
python "$STAGE1/sample_preferences.py" --input-dir "$PROCESSED/$DATASET" \
  --output "$STAGE1/data/scft/${SLUG}-scft.jsonl" --seed 0
```

**Output:**  A JSONL file with one record per positive or negative example. Each record contains `user_id`, `user_text`, `item`, and `label`. Step 4 pairs the positive with each negative to form training triplets.

```text
stage1_FT_ServiceLLM/data/scft/
└── <slug>-scft.jsonl
```

### Step 4. Fine-Tune ServiceLLM

**Input:**  The SCFT JSONL from Step 3 and the [Llama 3.1 8B base model](https://huggingface.co/meta-llama/Llama-3.1-8B). Set `BASE_MODEL` to a local checkpoint directory or to `meta-llama/Llama-3.1-8B` after configuring access.

```bash
BASE_MODEL=/path/to/Meta-Llama-3.1-8B

CUDA_VISIBLE_DEVICES=0 python "$STAGE1/main.py" \
  --base_model_name "$BASE_MODEL" --train_file "$STAGE1/data/scft/${SLUG}-scft.jsonl" \
  --output_dir "$STAGE1/checkpoints/${SLUG}-adapter" --seed 42 \
  --lora_r 16 --lora_alpha 32 --learning_rate 2e-4 --max_length 2048 \
  --per_device_train_batch_size 1 --gradient_accumulation_steps 8
```

**Output:**  The adapter configuration and weights saved after three epochs. Step 5 loads them alongside the same base model.

```text
stage1_FT_ServiceLLM/checkpoints/<slug>-adapter/
├── adapter_config.json
└── adapter_model.safetensors
```

### Step 5. Extract ServiceLLM Embeddings

**Input:**  `train.pkl`, `users.jsonl`, and `items.jsonl` from Step 2, plus the base model and trained adapter from Step 4.

```bash
CUDA_VISIBLE_DEVICES=0 python "$STAGE1/extract_embeddings.py" \
  --input-dir "$PROCESSED/$DATASET" --base-model "$BASE_MODEL" \
  --adapter "$STAGE1/checkpoints/${SLUG}-adapter" \
  --output "$STAGE1/data/emb/${SLUG}-4096.pt"
```

**Output:**  A tensor with one 4,096-dimensional vector per user or item. User prompts come from the training histories. User rows come first, followed by item rows, each in numeric ID order.

```text
stage1_FT_ServiceLLM/data/emb/
└── <slug>-4096.pt
```

### Step 6. Apply PCA and Prepare Recommendation Inputs

**Input:**  The 4,096-dimensional tensor from Step 5 and the train/test splits from Step 2.

```bash
python "$STAGE1/reduce_embeddings.py" --input "$STAGE1/data/emb/${SLUG}-4096.pt" \
  --output "$REC_DATA/emb/$DATASET/$SLUG.pt" --components 64 --seed 1234 &&
mkdir -p "$REC_DATA/$DATASET" &&
cp "$PROCESSED/$DATASET/train.pkl" "$PROCESSED/$DATASET/test.pkl" "$REC_DATA/$DATASET/"
```

**Output:**  The rebuilt recommendation inputs under `data/rebuilt/`, separate from the bundled Scientific files.

```text
stage2_Downstream_Recommendation/data/rebuilt/
├── <dataset>/
│   ├── train.pkl
│   └── test.pkl
└── emb/<dataset>/
    └── <slug>.pt     # 64-dimensional user and item embeddings
```

## Stage 2. Downstream Recommendation

### Step 7. Train LLGCN and InteRec and Evaluate

**Input:**  The train/test splits and 64-dimensional semantic tensor in the Stage 2 data directory. The graph uses the training interactions. The command below shows the Scientific settings. Choose a new output directory for each run.

```bash
CUDA_VISIBLE_DEVICES=0 python "$STAGE2/main.py" --dataset "$DATASET" \
  --data-root "$REC_DATA" --device cuda --seed 1234 \
  --num-layers 1 --margin 0.1 --batch-size 10000 --epochs 50 \
  --output-dir "$STAGE2/runs/${SLUG}-from-raw"
```

**Output:**  The recommendation checkpoint and evaluation metrics. Evaluation masks training items, ranks the full catalog, and reports Recall and NDCG.

```text
stage2_Downstream_Recommendation/runs/<slug>-from-raw/
├── model.pt
└── metrics.json
```

## Quick Start with the Scientific Dataset

**Input:**  The bundled Scientific train/test splits and post-PCA ServiceLLM embeddings from the original processed cache. Start directly at Stage 2.

```text
stage2_Downstream_Recommendation/data/
├── Scientific/
│   ├── train.pkl
│   ├── test.pkl
└── emb/Scientific/
    └── scientific.pt
```

### Run Stage 2

```bash
python "$STAGE2/main.py" --dataset Scientific --device cuda:0 \
  --seed 1234 --num-layers 1 --margin 0.1 \
  --batch-size 10000 --epochs 1000 --eval-freq 20 --output-dir "$STAGE2/runs/scientific-full"
```

**Output:** 

```text
stage2_Downstream_Recommendation/runs/scientific-full/
├── model.pt
└── metrics.json
```
