import os
import json
import math
import re
import time
from typing import List, Dict, Any, Optional

import torch
from transformers import AutoTokenizer, AutoModel
from vllm import LLM, SamplingParams

# ============================================================
# 1. CONFIG
# ============================================================
BASE_INPUT_FILE = r"/nvme1/benchmark/QDA_Benchmark/scripts/data/benchmark_theme.jsonl"
CODE_RESULT_FILE = r"/nvme1/benchmark/QDA_Benchmark/scripts/data/result/task2_result/14BQwen2.5/result_code_-ssd-amax-qda_models-Qwen2.5-14B-Instruct_n3.jsonl"
E2E_OUTPUT_FILE = r"/nvme1/benchmark/QDA_Benchmark/scripts/data/result/task2.5/result_e2e_14B_Qwen2,5_n3.jsonl"
EMBEDDING_MODEL_PATH = r"/nvme1/benchmark/QDA_Benchmark/scripts/model/all-MiniLM-L6-v2"
VLLM_MODEL_PATH = r"/ssd/amax/qda_models/Qwen2.5-14B-Instruct"

TENSOR_PARALLEL_SIZE = 8
N_RUNS = 3
BATCH_SIZE = 16              # If GPU utilization is still low, try 32.
MAX_TOKENS = 1024
TEMPERATURE = 0.7
TOP_P = 0.9

# Keep embedding on CPU by default: vLLM may reserve most GPU memory.
EMBED_BATCH_SIZE = 128

SUBTHEME_PROMPT = r"""
You are an expert in qualitative data analysis.

Your task is to group the provided codes into meaningful sub-themes.

A sub-theme should represent a meaningful conceptual grouping of related codes.

Important requirements:

1. Every provided code MUST be assigned to a sub-theme.
2. Do not omit any code.
3. Do not invent codes that are not present in the input.
4. A sub-theme can contain multiple related codes.
5. Give each sub-theme a concise and meaningful name.
6. Output ONLY valid JSON.
7. The JSON format must be:

{
  "subtheme": [
    "sub-theme 1",
    "sub-theme 2",
    "sub-theme 3"
  ]
}

Codes:
{inputData}
"""

THEME_PROMPT = r"""
You are an expert in qualitative data analysis.

Your task is to group the provided sub-themes into meaningful higher-level themes.

A theme should represent a meaningful conceptual grouping of related sub-themes.

Important requirements:

1. Every provided sub-theme MUST be represented.
2. Do not omit any sub-theme.
3. Do not invent sub-themes that are not present in the input.
4. Group related sub-themes into broader conceptual themes.
5. Give each theme a concise and meaningful name.
6. Output ONLY valid JSON.
7. The JSON format must be:

{
  "theme": [
    "theme 1",
    "theme 2",
    "theme 3"
  ]
}

Sub-Themes:
{inputData}
"""

# ============================================================
# JSONL / IDX
# ============================================================
def load_jsonl(path: str) -> List[Dict[str, Any]]:
    records = []
    if not os.path.exists(path):
        return records
    with open(path, "r", encoding="utf-8") as f:
        for line_number, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError as e:
                print(f"[WARNING] JSON error in {os.path.basename(path)} line {line_number}: {e}")
    return records


def append_jsonl(path: str, record: Dict[str, Any]):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")
        f.flush()
        os.fsync(f.fileno())


def normalize_idx(idx):
    if idx is None:
        return None
    if isinstance(idx, float) and idx.is_integer():
        return str(int(idx))
    return str(idx).strip()


def load_code_results(path: str) -> Dict[str, Dict[str, Any]]:
    out = {}
    for record in load_jsonl(path):
        idx = normalize_idx(record.get("idx"))
        if idx is not None:
            out[idx] = record
    print(f"[INFO] Loaded {len(out)} Code results.")
    return out


def load_existing_e2e(path: str) -> Dict[str, Dict[str, Any]]:
    """Load latest record for each idx. Existing output may contain duplicate idxs."""
    out = {}
    for record in load_jsonl(path):
        idx = normalize_idx(record.get("idx"))
        if idx is not None:
            out[idx] = record
    return out


def is_valid_result_dict(x):
    return isinstance(x, dict) and isinstance(x.get("best_pred"), list)


def is_theme_finished(record: Optional[Dict[str, Any]]) -> bool:
    if not record:
        return False
    return is_valid_result_dict(record.get("theme"))


def is_subtheme_finished(record: Optional[Dict[str, Any]]) -> bool:
    if not record:
        return False
    return is_valid_result_dict(record.get("subtheme"))

# ============================================================
# PROMPTS / PARSING
# ============================================================
def format_input_list(items: List[str]) -> str:
    return "\n".join(f"{i + 1}. {item}" for i, item in enumerate(items)) if items else ""


def build_subtheme_prompt(gen_code: List[str]) -> str:
    return SUBTHEME_PROMPT.replace("{inputData}", format_input_list(gen_code))


def build_theme_prompt(gen_subtheme: List[str]) -> str:
    return THEME_PROMPT.replace("{inputData}", format_input_list(gen_subtheme))


def parse_json_response(text: str) -> Optional[Dict[str, Any]]:
    text = text.strip()
    try:
        obj = json.loads(text)
        if isinstance(obj, dict):
            return obj
    except Exception:
        pass

    match = re.search(r"```(?:json)?\s*(.*?)\s*```", text, flags=re.DOTALL | re.IGNORECASE)
    if match:
        try:
            obj = json.loads(match.group(1).strip())
            if isinstance(obj, dict):
                return obj
        except Exception:
            pass

    start = text.find("{")
    end = text.rfind("}")
    if start != -1 and end > start:
        try:
            obj = json.loads(text[start:end + 1])
            if isinstance(obj, dict):
                return obj
        except Exception:
            pass
    return None


def extract_prediction(text: str, level: str) -> List[str]:
    obj = parse_json_response(text)
    if obj is None:
        return []
    pred = obj.get(level)
    if not isinstance(pred, list):
        return []
    return [x.strip() for x in pred if isinstance(x, str) and x.strip()]

# ============================================================
# EMBEDDING / METRICS
# ============================================================
def mean_pooling(model_output, attention_mask):
    token_embeddings = model_output[0]
    mask = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    return torch.sum(token_embeddings * mask, dim=1) / torch.clamp(mask.sum(dim=1), min=1e-9)


def encode_texts(texts: List[str], tokenizer, model) -> Optional[torch.Tensor]:
    if not texts:
        return None
    chunks = []
    with torch.no_grad():
        for start in range(0, len(texts), EMBED_BATCH_SIZE):
            batch = texts[start:start + EMBED_BATCH_SIZE]
            encoded = tokenizer(batch, padding=True, truncation=True, return_tensors="pt")
            output = model(**encoded)
            emb = mean_pooling(output, encoded["attention_mask"])
            chunks.append(torch.nn.functional.normalize(emb, p=2, dim=1))
    return torch.cat(chunks, dim=0)


def calculate_metrics(pred, gt, tokenizer, embedding_model):
    if not pred or not gt:
        return {"mms": 0.0, "precision_mms": 0.0, "recall_mms": 0.0, "f1_like": 0.0}

    # Encode each side ONCE, then reuse for precision and recall.
    emb_pred = encode_texts(pred, tokenizer, embedding_model)
    emb_gt = encode_texts(gt, tokenizer, embedding_model)
    sim = torch.mm(emb_pred, emb_gt.T)

    precision = sim.max(dim=1).values.mean().item()
    recall = sim.max(dim=0).values.mean().item()
    f1 = 0.0 if precision + recall <= 0 else 2 * precision * recall / (precision + recall)

    return {
        "mms": precision,
        "precision_mms": precision,
        "recall_mms": recall,
        "f1_like": f1,
    }

# ============================================================
# VLLM
# ============================================================
def create_llm():
    print("=" * 80)
    print("Loading LLM")
    print(f"Model: {VLLM_MODEL_PATH}")
    print(f"Tensor parallel: {TENSOR_PARALLEL_SIZE}")
    print(f"Batch size: {BATCH_SIZE}")
    print(f"n per prompt: {N_RUNS}")
    print(f"max_tokens: {MAX_TOKENS}")
    print("=" * 80)

    return LLM(
        model=VLLM_MODEL_PATH,
        tensor_parallel_size=TENSOR_PARALLEL_SIZE,
        dtype="auto",
        trust_remote_code=True,
    )


sampling_params = SamplingParams(
    temperature=TEMPERATURE,
    top_p=TOP_P,
    max_tokens=MAX_TOKENS,
    n=N_RUNS,
)


def generate_batch(llm, prompts: List[str], levels: List[str]):
    """One vLLM call for many prompts; each prompt produces N_RUNS samples."""
    outputs = llm.generate(prompts, sampling_params)
    results = []
    for request_output, level in zip(outputs, levels):
        runs = []
        for run_id, sample in enumerate(request_output.outputs[:N_RUNS], 1):
            text = sample.text
            pred = extract_prediction(text, level)
            runs.append({"run": run_id, "pred": pred, "raw_output": text})
        results.append(runs)
    return results

# ============================================================
# RECORD CONSTRUCTION
# ============================================================
def score_runs(runs, gt, tokenizer, embedding_model):
    for run in runs:
        metrics = calculate_metrics(run["pred"], gt, tokenizer, embedding_model)
        run.update(metrics)
    best = max(runs, key=lambda x: x["mms"]) if runs else {
        "run": 0, "pred": [], "mms": 0.0,
        "precision_mms": 0.0, "recall_mms": 0.0,
        "f1_like": 0.0, "raw_output": ""
    }
    return {
        "best_pred": best["pred"],
        "best_mms": best["mms"],
        "best_precision_mms": best["precision_mms"],
        "best_recall_mms": best["recall_mms"],
        "best_f1_like": best["f1_like"],
        "best_run": best["run"],
        "all_runs": runs,
    }


def base_record_template(base_record, gen_code, gt_subtheme, gt_theme):
    return {
        "idx": base_record.get("idx"),
        "domain": base_record.get("domain"),
        "research_question": base_record.get("research_question"),
        "gen_code": gen_code,
        "code_best_mms": None,
        "gt_subtheme": gt_subtheme,
        "subtheme": None,
        "gt_theme": gt_theme,
        "theme": None,
    }

# ============================================================
# BATCH PIPELINE
# ============================================================
def run_stage_batch(
    llm,
    jobs: List[Dict[str, Any]],
    level: str,
    tokenizer,
    embedding_model,
):
    prompts = [job["prompt"] for job in jobs]
    levels = [level] * len(jobs)
    all_runs = generate_batch(llm, prompts, levels)

    for job, runs in zip(jobs, all_runs):
        gt = job["gt"]
        scored = score_runs(runs, gt, tokenizer, embedding_model)
        job["result"] = scored
    return jobs


def process_batch(llm, batch, existing, code_results, tokenizer, embedding_model):
    """Process one batch while preserving old completed results."""
    # --------------------------
    # STEP 1: Sub-theme
    # --------------------------
    sub_jobs = []
    ready_for_theme = []

    for item in batch:
        idx = item["idx"]
        old = existing.get(idx)

        if is_subtheme_finished(old):
            print(f"  [RESUME] idx={idx}: Sub-theme already finished.")
            ready_for_theme.append((item, old["subtheme"]))
        else:
            sub_jobs.append({
                "idx": idx,
                "prompt": build_subtheme_prompt(item["gen_code"]),
                "gt": item["gt_subtheme"],
                "item": item,
            })

    if sub_jobs:
        print(f"  [SUBTHEME] generating {len(sub_jobs)} prompts × {N_RUNS} runs")
        t0 = time.time()
        run_stage_batch(llm, sub_jobs, "subtheme", tokenizer, embedding_model)
        print(f"  [SUBTHEME] batch finished in {time.time() - t0:.1f}s")

        for job in sub_jobs:
            item = job["item"]
            idx = item["idx"]
            result = job["result"]
            ready_for_theme.append((item, result))

            # IMPORTANT: checkpoint Sub-theme immediately.
            record = base_record_template(
                item["base"], item["gen_code"], item["gt_subtheme"], item["gt_theme"]
            )
            record["code_best_mms"] = item["code_best_mms"]
            record["subtheme"] = result
            record["theme"] = None
            append_jsonl(E2E_OUTPUT_FILE, record)
            existing[idx] = record
            print(f"    [CHECKPOINT] idx={idx} Sub-theme saved; best MMS={result['best_mms']:.4f}")

    # --------------------------
    # STEP 2: Theme
    # --------------------------
    theme_jobs = []
    for item, sub_result in ready_for_theme:
        idx = item["idx"]
        old = existing.get(idx)

        if is_theme_finished(old):
            print(f"  [RESUME] idx={idx}: Theme already finished.")
            continue

        gen_subtheme = sub_result.get("best_pred", [])
        theme_jobs.append({
            "idx": idx,
            "prompt": build_theme_prompt(gen_subtheme),
            "gt": item["gt_theme"],
            "item": item,
            "sub_result": sub_result,
        })

    if theme_jobs:
        print(f"  [THEME] generating {len(theme_jobs)} prompts × {N_RUNS} runs")
        t0 = time.time()
        run_stage_batch(llm, theme_jobs, "theme", tokenizer, embedding_model)
        print(f"  [THEME] batch finished in {time.time() - t0:.1f}s")

        for job in theme_jobs:
            item = job["item"]
            idx = item["idx"]
            theme_result = job["result"]
            sub_result = job["sub_result"]

            record = base_record_template(
                item["base"], item["gen_code"], item["gt_subtheme"], item["gt_theme"]
            )
            record["code_best_mms"] = item["code_best_mms"]
            record["subtheme"] = sub_result
            record["theme"] = theme_result
            append_jsonl(E2E_OUTPUT_FILE, record)
            existing[idx] = record
            print(f"    [SAVED] idx={idx} Theme best MMS={theme_result['best_mms']:.4f}")

# ============================================================
# STATISTICS FROM SAVED METRICS (NO RE-ENCODING)
# ============================================================
def calculate_e2e_statistics_from_records(records):
    sub_p_scores, sub_r_scores, sub_f1_scores = [], [], []
    theme_p_scores, theme_r_scores, theme_f1_scores = [], [], []

    for record in records:
        sub = record.get("subtheme")
        theme = record.get("theme")
        if not isinstance(sub, dict) or not isinstance(theme, dict):
            continue
        sr = sub.get("all_runs", [])
        tr = theme.get("all_runs", [])
        if len(sr) != N_RUNS or len(tr) != N_RUNS:
            continue

        sub_p_scores.append([x.get("precision_mms", 0.0) for x in sr])
        sub_r_scores.append([x.get("recall_mms", 0.0) for x in sr])
        sub_f1_scores.append([x.get("f1_like", 0.0) for x in sr])
        theme_p_scores.append([x.get("precision_mms", 0.0) for x in tr])
        theme_r_scores.append([x.get("recall_mms", 0.0) for x in tr])
        theme_f1_scores.append([x.get("f1_like", 0.0) for x in tr])

    def summary(scores):
        if not scores:
            return {"run1": 0, "run2": 0, "run3": 0, "mean": 0, "std": 0, "best": 0}
        run_means = [sum(x[i] for x in scores) / len(scores) for i in range(N_RUNS)]
        mean = sum(run_means) / N_RUNS
        std = math.sqrt(sum((x - mean) ** 2 for x in run_means) / N_RUNS)
        best = sum(max(x) for x in scores) / len(scores)
        return {"run1": run_means[0], "run2": run_means[1], "run3": run_means[2],
                "mean": mean, "std": std, "best": best}

    return {
        "entries": len(sub_p_scores),
        "subtheme": {
            "precision_mms": summary(sub_p_scores),
            "recall_mms": summary(sub_r_scores),
            "f1_like": summary(sub_f1_scores),
        },
        "theme": {
            "precision_mms": summary(theme_p_scores),
            "recall_mms": summary(theme_r_scores),
            "f1_like": summary(theme_f1_scores),
        },
    }


def print_statistics(stats):
    print("\n" + "=" * 80)
    print("E2E STATISTICS")
    print("=" * 80)
    for level in ["subtheme", "theme"]:
        print(f"\n{level.upper()}")
        for metric in ["precision_mms", "recall_mms", "f1_like"]:
            s = stats[level][metric]
            print(f"{metric:16s}: {s['mean']*100:.2f} ± {s['std']*100:.2f} | Best {s['best']*100:.2f}")

# ============================================================
# MAIN
# ============================================================
def main():
    print("=" * 80)
    print("FAST E2E QDA PIPELINE")
    print("Resume-compatible with existing E2E JSONL")
    print("=" * 80)

    if not os.path.exists(BASE_INPUT_FILE):
        raise FileNotFoundError(BASE_INPUT_FILE)
    if not os.path.exists(CODE_RESULT_FILE):
        raise FileNotFoundError(CODE_RESULT_FILE)

    base_records = load_jsonl(BASE_INPUT_FILE)
    code_results = load_code_results(CODE_RESULT_FILE)
    existing = load_existing_e2e(E2E_OUTPUT_FILE)

    print(f"[INFO] Base records: {len(base_records)}")
    print(f"[INFO] Existing E2E idxs: {len(existing)}")
    print(f"[INFO] Already Theme-finished: {sum(is_theme_finished(x) for x in existing.values())}")
    print(f"[INFO] Sub-theme checkpoints: {sum(is_subtheme_finished(x) and not is_theme_finished(x) for x in existing.values())}")

    # Build work items. Finished Theme records are skipped; Sub-theme checkpoints are resumed.
    work = []
    skipped = 0
    missing_code = 0

    for base_record in base_records:
        idx = normalize_idx(base_record.get("idx"))
        if idx is None:
            skipped += 1
            continue

        old = existing.get(idx)
        if is_theme_finished(old):
            skipped += 1
            continue

        code_record = code_results.get(idx)
        if code_record is None:
            print(f"[WARNING] No Code result for idx={idx}")
            missing_code += 1
            continue

        gen_code = code_record.get("best_pred")
        if not isinstance(gen_code, list):
            br = code_record.get("best_run")
            if isinstance(br, dict) and isinstance(br.get("pred"), list):
                gen_code = br["pred"]
        if not gen_code:
            print(f"[WARNING] Empty Code best_pred for idx={idx}")
            missing_code += 1
            continue

        gt_subtheme = base_record.get("gt_subtheme", [])
        gt_theme = base_record.get("gt_theme", [])
        if not isinstance(gt_subtheme, list):
            gt_subtheme = []
        if not isinstance(gt_theme, list):
            gt_theme = []

        work.append({
            "idx": idx,
            "base": base_record,
            "gen_code": gen_code,
            "code_best_mms": code_record.get("best_mms"),
            "gt_subtheme": gt_subtheme,
            "gt_theme": gt_theme,
        })

    print(f"[INFO] Remaining work: {len(work)}")
    print(f"[INFO] Batch size: {BATCH_SIZE}")

    if not work:
        print("[INFO] Nothing to run.")
        if os.path.exists(E2E_OUTPUT_FILE):
            records = load_jsonl(E2E_OUTPUT_FILE)
            stats = calculate_e2e_statistics_from_records(records)
            print_statistics(stats)
        return

    print("\nLoading embedding model...")
    tokenizer = AutoTokenizer.from_pretrained(EMBEDDING_MODEL_PATH)
    embedding_model = AutoModel.from_pretrained(EMBEDDING_MODEL_PATH)
    embedding_model.eval()

    llm = create_llm()

    start_all = time.time()
    try:
        for start in range(0, len(work), BATCH_SIZE):
            batch = work[start:start + BATCH_SIZE]
            print("\n" + "=" * 80)
            print(f"BATCH {start // BATCH_SIZE + 1} | items {start + 1}-{start + len(batch)} / {len(work)}")
            print("idxs:", ", ".join(x["idx"] for x in batch))
            print("=" * 80)
            process_batch(llm, batch, existing, code_results, tokenizer, embedding_model)

            elapsed = time.time() - start_all
            done = min(start + len(batch), len(work))
            print(f"[PROGRESS] {done}/{len(work)} remaining-work items; elapsed={elapsed/60:.1f} min")

    except KeyboardInterrupt:
        print("\n[INTERRUPTED] Safe to stop. Completed Theme results and Sub-theme checkpoints were flushed to JSONL.")
        print("[RESUME] Run this same script again; it will skip completed Theme records and continue from Sub-theme checkpoints.")
        return

    # Statistics use saved metrics only; no MiniLM recomputation.
    records = load_jsonl(E2E_OUTPUT_FILE)
    stats = calculate_e2e_statistics_from_records(records)
    print_statistics(stats)

    stats_file = os.path.splitext(E2E_OUTPUT_FILE)[0] + "_statistics.json"
    with open(stats_file, "w", encoding="utf-8") as f:
        json.dump(stats, f, ensure_ascii=False, indent=2)

    print(f"\n[SAVED] Statistics: {stats_file}")
    print(f"[DONE] Total wall time: {(time.time() - start_all)/60:.1f} min")
    print(f"[DONE] Skipped existing finished: {skipped}")
    print(f"[DONE] Missing Code: {missing_code}")


if __name__ == "__main__":
    main()
