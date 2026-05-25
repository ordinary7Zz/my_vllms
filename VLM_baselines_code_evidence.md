# VLM baselines in current codebase: MedGemma and Qwen3

This note summarizes the current code-backed facts for the two VLM baselines used for thyroid malignancy classification experiments. All statements below are based only on code, scripts, configuration, and visible comments in this repository. If a fact is not directly supported by code, it is marked as **Not found in code**.

## A. Executive summary

- **MedGemma and Qwen3 are not used the same way as the GPT baseline in the current code.** GPT uses API-based prompt inference and parses returned JSON confidence, while MedGemma/Qwen3 have both zero-shot eval scripts and **separate LoRA fine-tuning scripts**. Evidence: `gpt/gpt_thyroid_binary_eval.py:79-117`, `medgemma/medgemma_thyroid_binary_train.py:27-56`, `qwen3/qwen3_vl_thyroid_binary_train.py:24-54`.
- The **explicit checkpoint string** for Qwen3 is found in code as **`Qwen/Qwen3-VL-8B-Instruct`**. Evidence: `download_qwen3_vl_8b_instruct_modelscope.py:7-8`.
- The **explicit MedGemma Hugging Face / ModelScope identifier is not found in code**; only a local directory example **`/mnt/wangbd8/workspace/medgemma-4b-it`** is referenced. Evidence: `my_run.md:89-93`, `medgemma/medgemma_thyroid_binary_eval.py:135-137`.
- The current MedGemma/Qwen3 training code is **LoRA / optional QLoRA causal-LM fine-tuning**, not full fine-tuning, not classifier-head-only training, and not adapter tuning beyond PEFT LoRA adapters. Evidence: `common/vlm_sft.py:42-50`, `common/vlm_sft.py:53-90`.
- **Both models use the same training paradigm**: next-token supervision on the textual answer `" 0"` or `" 1"` appended to an image-conditioned prompt/chat template. Evidence: `common/vlm_sft.py:99-145`, `common/thyroid_data.py:105`, `common/thyroid_prompts.py:3-10`, `common/thyroid_prompts.py:12-23`.
- Their task formulation is **image + text prompt**, not image-only. Evidence: MedGemma prompt `common/thyroid_prompts.py:3-10`; Qwen3 chat template `common/thyroid_prompts.py:12-23`, `common/thyroid_prompts.py:30-43`.
- Their inference is **generative scoring-based binary classification**, not a discriminative classifier emitting a 2-dim classification head. The scripts compute **log-probabilities of candidate answer strings** (`"0"` vs `"1"` variants) and convert them to `P(malignant)`. Evidence: `medgemma/medgemma_thyroid_binary_eval.py:94-127`, `qwen3/qwen3_vl_thyroid_binary_eval.py:105-147`.
- **AUROC/AUPRC are computed from `p_malignant`**, and the hard label is obtained by thresholding at 0.5 by default. Evidence: `medgemma/medgemma_thyroid_binary_eval.py:149-158`, `medgemma/medgemma_thyroid_binary_eval.py:238-250`; `qwen3/qwen3_vl_thyroid_binary_eval.py:177-186`, `qwen3/qwen3_vl_thyroid_binary_eval.py:253-269`; `common/thyroid_metrics.py:10-29`.
- For the **paper-result entrypoint currently documented in repo**, the zero-shot eval shell scripts point to **DDTI_Classification test labels** for both MedGemma and Qwen3. Evidence: `medgemma/medgemma_thyroid_binary_eval.sh:1-6`, `qwen3/qwen3_vl_thyroid_binary_eval.sh:1-6`.
- **Patient-level split, image resolution, optimizer type, scheduler type, and external extra preprocessing are not found in code.**

## B. Evidence table

| Topic | Finding | Evidence |
|---|---|---|
| Qwen3 checkpoint | Exact checkpoint string in code is `Qwen/Qwen3-VL-8B-Instruct` | `download_qwen3_vl_8b_instruct_modelscope.py:7-8` |
| MedGemma checkpoint | Exact remote checkpoint identifier not found; only local path `medgemma-4b-it` appears | `my_run.md:89-93`, `medgemma/medgemma_thyroid_binary_eval.py:135-137` |
| Paper wording vs Qwen3 name | “Qwen3-vl-8B” is imprecise relative to code; code uses `Qwen3-VL-8B-Instruct` | `download_qwen3_vl_8b_instruct_modelscope.py:7-8`, `qwen3/qwen3_vl_thyroid_binary_eval.py:158-160` |
| Paper wording vs MedGemma name | “MedGemma” alone is less specific than code; code indicates a 4B instruction-tuned local model dir, but exact official ID is not found | `my_run.md:89-93`, `medgemma/medgemma_thyroid_binary_eval.py:135-137` |
| Training paradigm | Uses PEFT LoRA adapters; optional QLoRA via 4-bit BitsAndBytes | `common/vlm_sft.py:42-50`, `common/vlm_sft.py:53-90` |
| MedGemma train mode | MedGemma train script calls `load_model_with_lora(...)` | `medgemma/medgemma_thyroid_binary_train.py:66-76` |
| Qwen3 train mode | Qwen3 train script calls `load_model_with_lora(...)` | `qwen3/qwen3_vl_thyroid_binary_train.py:65-76` |
| Same paradigm? | Yes, both use same LoRA SFT scaffold and same label format | `common/vlm_sft.py:99-145`, `common/thyroid_data.py:90-111` |
| Input form: MedGemma | image + text prompt | `common/thyroid_prompts.py:3-10`, `common/vlm_sft.py:100-105`, `medgemma/medgemma_thyroid_binary_eval.py:110-118` |
| Input form: Qwen3 | image + chat text template | `common/thyroid_prompts.py:12-23`, `common/thyroid_prompts.py:30-43`, `common/vlm_sft.py:117-136` |
| Output type | No classification head logits are read; scripts score candidate generated strings | `medgemma/medgemma_thyroid_binary_eval.py:49-91`, `medgemma/medgemma_thyroid_binary_eval.py:94-127`, `qwen3/qwen3_vl_thyroid_binary_eval.py:50-86`, `qwen3/qwen3_vl_thyroid_binary_eval.py:105-147` |
| Score for AUROC/AUPRC | `y_prob` = `p_malignant` | `medgemma/medgemma_thyroid_binary_eval.py:241-250`, `qwen3/qwen3_vl_thyroid_binary_eval.py:256-269`, `common/thyroid_metrics.py:10-12` |
| Hard label derivation | `pred = 1 if p1 >= threshold else 0`, threshold default 0.5 | `medgemma/medgemma_thyroid_binary_eval.py:149-150`, `medgemma/medgemma_thyroid_binary_eval.py:238`; `qwen3/qwen3_vl_thyroid_binary_eval.py:177-178`, `qwen3/qwen3_vl_thyroid_binary_eval.py:253` |
| Zero-shot eval entrypoint | Both `.sh` files run base models on DDTI_Classification test labels without adapter | `medgemma/medgemma_thyroid_binary_eval.sh:1-6`, `qwen3/qwen3_vl_thyroid_binary_eval.sh:1-6` |
| Fine-tuned eval support | Both eval scripts can optionally load `--adapter_dir` | `medgemma/medgemma_thyroid_binary_eval.py:137-139`, `medgemma/medgemma_thyroid_binary_eval.py:175-185`; `qwen3/qwen3_vl_thyroid_binary_eval.py:162-166`, `qwen3/qwen3_vl_thyroid_binary_eval.py:205-220` |
| GPT baseline usage | GPT uses API prompt inference, returns JSON with prediction/confidence/reasoning | `gpt/gpt_thyroid_binary_eval.py:79-117`, `gpt/gpt_thyroid_binary_eval.py:135-176` |
| Train/test split evidence | Train scripts accept user-provided `train_json` and `test_json`; split policy itself not encoded | `medgemma/medgemma_thyroid_binary_train.py:29-33`, `qwen3/qwen3_vl_thyroid_binary_train.py:26-30` |
| Patient-level split | Not found in code | No direct evidence found |
| Image resolution | Not found in code | No resize/image-size args or transforms in searched files |
| Optimizer type | Not found explicitly in code | `common/vlm_sft.py:148-171` sets `TrainingArguments` but no explicit `optim` |
| Scheduler type | Not found explicitly in code | `common/vlm_sft.py:148-171` sets `warmup_ratio` only |
| Mixed precision | bf16/fp16/fp32 supported | `medgemma/medgemma_thyroid_binary_train.py:44`, `qwen3/qwen3_vl_thyroid_binary_train.py:41`, `common/vlm_sft.py:14-18`, `common/vlm_sft.py:165-166` |
| Quantization | Optional QLoRA 4-bit NF4 | `common/vlm_sft.py:42-50`, `common/vlm_sft.py:79-80` |
| Checkpoint selection | Optional `load_best_model_at_end`, metric `eval_auroc` | `common/vlm_sft.py:148-171` |
| Random seed | Default seed 42 in train and CI bootstrap | `medgemma/medgemma_thyroid_binary_train.py:43`, `qwen3/qwen3_vl_thyroid_binary_train.py:40`, `medgemma/medgemma_thyroid_binary_eval.py:157`, `qwen3/qwen3_vl_thyroid_binary_eval.py:185` |

## C. Per-model details

### MedGemma

#### Accurate model / checkpoint name
- **Code-supported name:** local model dir example is **`/mnt/wangbd8/workspace/medgemma-4b-it`**. Evidence: `my_run.md:89-93`, `medgemma/medgemma_thyroid_binary_eval.py:135-137`.
- **Specific official checkpoint ID:** **Not found in code.**
- **Consistency with current paper wording “MedGemma”:** directionally consistent, but less specific than current code.

#### How it is used for classification
- Dedicated training script loads the base model with **LoRA adapters**, via `load_model_with_lora(...)`. Evidence: `medgemma/medgemma_thyroid_binary_train.py:66-76`.
- `load_model_with_lora(...)` constructs a `LoraConfig` with `TaskType.CAUSAL_LM` and wraps the model using `get_peft_model(...)`. Evidence: `common/vlm_sft.py:81-90`.
- Optional **QLoRA** is supported via `--use_qlora`, which enables 4-bit NF4 quantization and `prepare_model_for_kbit_training(...)`. Evidence: `medgemma/medgemma_thyroid_binary_train.py:47`, `common/vlm_sft.py:42-50`, `common/vlm_sft.py:79-80`.
- Therefore: **not full fine-tuning**, **not classifier-head-only**, **yes: LoRA / optional QLoRA generative fine-tuning**.

#### Input form
- **Image + text prompt**, not image-only. Evidence: `common/thyroid_prompts.py:3-10`, `common/vlm_sft.py:100-105`, `medgemma/medgemma_thyroid_binary_eval.py:110`.
- Training prompt literal:

```text
<start_of_image>
You are a medical imaging assistant.
Task: Thyroid ultrasound nodule malignancy classification.
Output exactly one character: 0 or 1.
0 = benign, 1 = malignant.
Answer:
```

Evidence: `common/thyroid_prompts.py:3-10`

- During training, the answer text is appended as `" 0"` or `" 1"` to that prefix. Evidence: `common/thyroid_data.py:105`, `common/vlm_sft.py:101-104`.
- During inference, the same prompt prefix is used and the script scores candidate answer strings. Evidence: `medgemma/medgemma_thyroid_binary_eval.py:188`, `medgemma/medgemma_thyroid_binary_eval.py:221-236`.
- This is **generative binary classification**, not discriminative classification with a dedicated classification head.

#### Output form
- The eval script computes **log-probabilities of candidate output strings** `" 0"`/`" 1"` (or fallback `"0"`/`"1"`). Evidence: `medgemma/medgemma_thyroid_binary_eval.py:114-127`, `medgemma/medgemma_thyroid_binary_eval.py:220-236`.
- Output type used by evaluation:
  - underlying model output: token logits from autoregressive LM forward pass
  - derived score: `p_malignant = exp(lp1) / (exp(lp0)+exp(lp1))`
- Final label: `pred = 1 if p1 >= threshold else 0`, default threshold `0.5`. Evidence: `medgemma/medgemma_thyroid_binary_eval.py:149-150`, `medgemma/medgemma_thyroid_binary_eval.py:238`.
- AUROC/AUPRC use `y_prob = p_malignant`. Evidence: `medgemma/medgemma_thyroid_binary_eval.py:240-250`, `common/thyroid_metrics.py:10-12`.

#### Training data and split
- Training script takes `--train_json` and `--test_json`; split is supplied externally, not encoded. Evidence: `medgemma/medgemma_thyroid_binary_train.py:29-33`.
- Example fine-tuning command in repo uses:
  - image dir: `.../train_val_test/Superimposed_multitask/dataset_3/train/images`
  - train labels: `dataset_3_train_label.json`
  - test labels: `dataset_3_test_label.json`
  Evidence: `my_run.md:155-160`
- Example zero-shot/paper-facing eval command uses:
  - DDTI_Classification `all/images`
  - `DDTI_Classification_test_label.json`
  Evidence: `medgemma/medgemma_thyroid_binary_eval.sh:1-6`, `my_run.md:119-128`
- Whether this split is the **same as other baselines**: **Not fully verifiable from code alone.**
- Whether it is **patient-level split**: **Not found in code.**
- External extra data / extra preprocessing: no direct evidence beyond alternate dataset paths (`dataset_3`, `TN3K`, `DDTI_Classification`).

#### Training configuration
- epochs: `3` — `medgemma/medgemma_thyroid_binary_train.py:34`
- per-device train batch size: `1` — `medgemma/medgemma_thyroid_binary_train.py:35`
- per-device eval batch size: `1` — `medgemma/medgemma_thyroid_binary_train.py:36`
- gradient accumulation: `8` — `medgemma/medgemma_thyroid_binary_train.py:37`
- learning rate: `2e-4` — `medgemma/medgemma_thyroid_binary_train.py:38`
- weight decay: `0.0` — `medgemma/medgemma_thyroid_binary_train.py:39`
- warmup ratio: `0.03` — `medgemma/medgemma_thyroid_binary_train.py:40`
- logging steps: `10` — `medgemma/medgemma_thyroid_binary_train.py:41`
- seed: `42` — `medgemma/medgemma_thyroid_binary_train.py:43`
- dtype choices: `bf16/fp16/fp32` — `medgemma/medgemma_thyroid_binary_train.py:44`
- num_workers: `0` — `medgemma/medgemma_thyroid_binary_train.py:45`
- gradient checkpointing: optional flag — `medgemma/medgemma_thyroid_binary_train.py:46`
- QLoRA: optional flag — `medgemma/medgemma_thyroid_binary_train.py:47`
- LoRA hyperparameters: `r=16`, `alpha=32`, `dropout=0.05` — `medgemma/medgemma_thyroid_binary_train.py:48-50`
- target modules: `q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj` — `medgemma/medgemma_thyroid_binary_train.py:51`
- eval/save strategy: `epoch` / `epoch` — `medgemma/medgemma_thyroid_binary_train.py:52-53`
- save total limit: `2` — `medgemma/medgemma_thyroid_binary_train.py:54`
- load best model at end: optional flag — `medgemma/medgemma_thyroid_binary_train.py:55`
- best-model metric: `eval_auroc` — `common/vlm_sft.py:148-171`
- optimizer: **Not found in code**
- scheduler type: **Not found in code**
- image resolution: **Not found in code**

#### Does it still use prompt-based inference after training?
- **Yes.** Even when `--adapter_dir` is supplied, the eval script still uses the prompt and scores `"0"` vs `"1"` candidate strings. Evidence: `medgemma/medgemma_thyroid_binary_eval.py:137-139`, `medgemma/medgemma_thyroid_binary_eval.py:175-185`, `medgemma/medgemma_thyroid_binary_eval.py:188`, `medgemma/medgemma_thyroid_binary_eval.py:221-236`.
- So after fine-tuning, it is **not converted into a separate discriminative classification head**.
- Relative to GPT:
  - GPT is **API prompt inference with free-form generated JSON and confidence extraction**. Evidence: `gpt/gpt_thyroid_binary_eval.py:104-117`, `gpt/gpt_thyroid_binary_eval.py:135-176`.
  - MedGemma is **locally run, optionally LoRA-fine-tuned, and evaluated by candidate log-prob scoring under a fixed binary prompt**.

### Qwen3

#### Accurate model / checkpoint name
- Exact checkpoint string found in code: **`Qwen/Qwen3-VL-8B-Instruct`**. Evidence: `download_qwen3_vl_8b_instruct_modelscope.py:7-8`.
- Eval script help/defaults also refer to **Qwen3-VL-8B-Instruct**. Evidence: `qwen3/qwen3_vl_thyroid_binary_eval.py:156-160`, `qwen3/qwen3_vl_thyroid_binary_eval.py:274`.
- Consistency with paper wording “Qwen3-vl-8B”: **not fully accurate**.

#### How it is used for classification
- Dedicated train script loads `Qwen3VLForConditionalGeneration` with **LoRA adapters**. Evidence: `qwen3/qwen3_vl_thyroid_binary_train.py:10`, `qwen3/qwen3_vl_thyroid_binary_train.py:65-76`.
- Optional QLoRA via `--use_qlora`. Evidence: `qwen3/qwen3_vl_thyroid_binary_train.py:44`, `common/vlm_sft.py:42-50`, `common/vlm_sft.py:79-80`.
- Therefore: **not full fine-tuning**, **not classifier-head-only**, **yes: LoRA / optional QLoRA generative fine-tuning**.

#### Input form
- **Image + text/chat template**, not image-only. Evidence: `common/thyroid_prompts.py:12-23`, `common/thyroid_prompts.py:30-43`, `qwen3/qwen3_vl_thyroid_binary_eval.py:89-102`.
- Training/inference template:
  - system: `You are a medical imaging assistant. You will be given a thyroid ultrasound image. Your task is binary malignancy classification.`
  - user text:

```text
Task: Thyroid ultrasound nodule malignancy classification.
Output exactly one character: 0 or 1.
0 = benign, 1 = malignant.
Answer:
```

Evidence: `common/thyroid_prompts.py:12-23`

- During training, assistant answer is appended as `" 0"` or `" 1"`. Evidence: `common/thyroid_prompts.py:41-42`, `common/thyroid_data.py:105`, `common/vlm_sft.py:117-145`.
- This is also **generative binary classification**, not discriminative head-based classification.

#### Output form
- Eval computes log-probabilities for candidate strings such as `(" 0"," 1")`, `("0","1")`, `("\n0","\n1")`, `(" 0\n"," 1\n")`. Evidence: `qwen3/qwen3_vl_thyroid_binary_eval.py:124-145`.
- Derived score is `p_malignant`. Evidence: `qwen3/qwen3_vl_thyroid_binary_eval.py:137-142`.
- Final label is thresholded at 0.5 by default. Evidence: `qwen3/qwen3_vl_thyroid_binary_eval.py:177-178`, `qwen3/qwen3_vl_thyroid_binary_eval.py:253`.
- AUROC/AUPRC use `y_prob = p_malignant`. Evidence: `qwen3/qwen3_vl_thyroid_binary_eval.py:255-269`, `common/thyroid_metrics.py:10-12`.

#### Training data and split
- Train script accepts external `train_json` / `test_json`. Evidence: `qwen3/qwen3_vl_thyroid_binary_train.py:26-30`.
- Example fine-tuning command uses TN3K:
  - image dir: `.../train_val_test/TN3K/images`
  - train labels: `my_json/tn3k_train_label.json`
  - test labels: `my_json/tn3k_test_label.json`
  Evidence: `my_run.md:187-193`
- Example zero-shot/paper-facing eval command uses DDTI_Classification test labels. Evidence: `qwen3/qwen3_vl_thyroid_binary_eval.sh:1-6`, `my_run.md:136-145`
- `my_json/tn3k_test_label.json` contains only `filename` and `malignancy`; no patient metadata. Evidence: `my_json/tn3k_test_label.json:1-2458`
- Patient-level split: **Not found in code**
- Extra external data / preprocessing: **Not found in code**, beyond alternate dataset paths.

#### Training configuration
- epochs: `3` — `qwen3/qwen3_vl_thyroid_binary_train.py:31`
- per-device train batch size: `1` — `qwen3/qwen3_vl_thyroid_binary_train.py:32`
- per-device eval batch size: `1` — `qwen3/qwen3_vl_thyroid_binary_train.py:33`
- gradient accumulation: `8` — `qwen3/qwen3_vl_thyroid_binary_train.py:34`
- learning rate: `2e-4` — `qwen3/qwen3_vl_thyroid_binary_train.py:35`
- weight decay: `0.0` — `qwen3/qwen3_vl_thyroid_binary_train.py:36`
- warmup ratio: `0.03` — `qwen3/qwen3_vl_thyroid_binary_train.py:37`
- logging steps: `10` — `qwen3/qwen3_vl_thyroid_binary_train.py:38`
- seed: `42` — `qwen3/qwen3_vl_thyroid_binary_train.py:40`
- dtype choices: `bf16/fp16/fp32` — `qwen3/qwen3_vl_thyroid_binary_train.py:41`
- gradient checkpointing optional — `qwen3/qwen3_vl_thyroid_binary_train.py:43`
- QLoRA optional — `qwen3/qwen3_vl_thyroid_binary_train.py:44`
- LoRA hyperparameters: `r=16`, `alpha=32`, `dropout=0.05` — `qwen3/qwen3_vl_thyroid_binary_train.py:45-47`
- target modules: `q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj` — `qwen3/qwen3_vl_thyroid_binary_train.py:48`
- eval/save strategy: `epoch` / `epoch` — `qwen3/qwen3_vl_thyroid_binary_train.py:49-50`
- save total limit: `2` — `qwen3/qwen3_vl_thyroid_binary_train.py:51`
- load best model at end: optional — `qwen3/qwen3_vl_thyroid_binary_train.py:52`
- attention impl flag: `auto|flash_attention_2|sdpa|eager` — `qwen3/qwen3_vl_thyroid_binary_train.py:53`
- best-model metric: `eval_auroc` — `common/vlm_sft.py:148-171`
- optimizer: **Not found in code**
- scheduler type: **Not found in code**
- image resolution: **Not found in code**

#### Does it still use prompt-based inference after training?
- **Yes.** Even with `--adapter_dir`, the eval path still uses the multimodal prompt/chat template and compares candidate answer-string log-probabilities. Evidence: `qwen3/qwen3_vl_thyroid_binary_eval.py:162-166`, `qwen3/qwen3_vl_thyroid_binary_eval.py:205-220`, `qwen3/qwen3_vl_thyroid_binary_eval.py:89-102`, `qwen3/qwen3_vl_thyroid_binary_eval.py:121-147`.
- Thus after fine-tuning, Qwen3 is still evaluated through **prompt-conditioned token scoring**, not via a separate discriminative classifier head.
- Relative to GPT:
  - GPT uses an external API and parses generated JSON confidence. Evidence: `gpt/gpt_thyroid_binary_eval.py:104-117`, `gpt/gpt_thyroid_binary_eval.py:165-181`.
  - Qwen3 is local, optionally LoRA-fine-tuned, and evaluated by direct conditional likelihood comparison of binary answers.

## D. Sentences in the paper that should be changed

- **Original**  
  “For VLM baselines, we use prompt-based inference rather than task-specific fine-tuning...”

  **Why inaccurate**  
  This is inaccurate if it refers to **all VLM baselines**. MedGemma and Qwen3 both have explicit task-specific fine-tuning scripts in the current codebase, implemented as LoRA/QLoRA. Evidence: `medgemma/medgemma_thyroid_binary_train.py:27-56`, `qwen3/qwen3_vl_thyroid_binary_train.py:24-54`, `common/vlm_sft.py:53-90`.

  **Suggested rewrite**  
  “For proprietary GPT baselines, we use prompt-based inference without task-specific fine-tuning. For the open-source MedGemma and Qwen3-VL baselines, the codebase also supports task-specific LoRA-based fine-tuning for binary malignancy classification.”

- **Original**  
  “All VLMs are evaluated using a unified binary malignancy-classification prompt with a single-character output format.”

  **Why inaccurate**  
  This is broadly true for **MedGemma/Qwen3**, but not for **GPT**. The GPT script requests a JSON response with `prediction`, `confidence`, and `reasoning`, not a single-character output. Evidence: `common/thyroid_prompts.py:3-10`, `common/thyroid_prompts.py:12-23`, `gpt/gpt_thyroid_binary_eval.py:104-117`.

  **Suggested rewrite**  
  “MedGemma and Qwen3-VL are evaluated with a binary prompt/template that constrains the target answer to `0` or `1`. In contrast, the GPT baseline is queried with a JSON-formatted prompt that returns a binary prediction together with a confidence score and brief reasoning.”

- **Original**  
  Any wording implying that MedGemma/Qwen3 and GPT use the same inference pipeline.

  **Why inaccurate**  
  Current code shows a clear difference:
  1. GPT: API generation + parsed JSON confidence.
  2. MedGemma/Qwen3: local models; optional LoRA adapters; conditional log-probability scoring of binary answer strings.
  Evidence: `gpt/gpt_thyroid_binary_eval.py:79-117`, `gpt/gpt_thyroid_binary_eval.py:135-176`, `medgemma/medgemma_thyroid_binary_eval.py:94-127`, `qwen3/qwen3_vl_thyroid_binary_eval.py:105-147`.

  **Suggested rewrite**  
  “GPT-5.1-style baselines and the open-source VLM baselines are not used identically in our codebase: the GPT baseline is evaluated via API prompting, whereas MedGemma and Qwen3-VL are local vision-language models that can be optionally LoRA-fine-tuned and are scored by the conditional likelihood of binary answer tokens.”

- **Original**  
  “Qwen3-vl-8B”

  **Why inaccurate**  
  The code explicitly identifies the checkpoint as `Qwen/Qwen3-VL-8B-Instruct`. Evidence: `download_qwen3_vl_8b_instruct_modelscope.py:7-8`.

  **Suggested rewrite**  
  “Qwen3-VL-8B-Instruct”

- **Original**  
  Any wording that presents MedGemma as a precisely specified official checkpoint.

  **Why inaccurate**  
  The current released code only supports the local directory name `medgemma-4b-it`; it does not expose a more specific official hub identifier.

  **Suggested rewrite**  
  “a local MedGemma 4B instruction-tuned checkpoint (`medgemma-4b-it` in the codebase)”

## E. Manuscript-ready English paragraph

In the current codebase, the proprietary GPT baseline is evaluated by direct prompt-based API inference, where the model returns a JSON response containing a binary prediction and a confidence score. By contrast, the MedGemma and Qwen3-VL baselines are implemented as local vision-language models with separate task-specific fine-tuning scripts based on PEFT LoRA, with optional 4-bit QLoRA support. For both MedGemma and Qwen3-VL, the classification task is formulated as image-conditioned text generation with a binary target output (`0` for benign and `1` for malignant). At evaluation time, these models are scored by comparing the conditional log-probabilities of the candidate answers `0` and `1`, and the resulting normalized probability of the malignant label is used for thresholded prediction as well as AUROC and AUPRC computation. The code explicitly identifies the Qwen baseline as **Qwen3-VL-8B-Instruct**, whereas the MedGemma code references a local **medgemma-4b-it** checkpoint path but does not expose a more specific official checkpoint identifier.
