"""
============================================================
Task 2 - Precision-MMS / Recall-MMS / F1-like 批量统计
============================================================

目录结构：

task2_result/
│
├── calc_precision_mms.py
│
├── task2_gpt5/
│   ├── result_code_gpt-5_n3.jsonl
│   ├── result_subtheme_gpt-5_n3.jsonl
│   └── result_theme_gpt-5_n3.jsonl
│
├── task2_qwen37max/
│   ├── result_code_qwen3.7-max_n3.jsonl
│   ├── result_subtheme_qwen3.7-max_n3.jsonl
│   └── result_theme_qwen3.7-max_n3.jsonl
│
└── ...

直接运行：

    python calc_precision_mms.py

程序会自动：

1. 识别所有模型文件夹
2. 识别 code / subtheme / theme
3. 读取每个文件的三次生成结果
4. 分别计算 Precision-MMS
5. 分别计算 Recall-MMS
6. 分别计算 F1-like
7. 计算 Mean ± Std
8. 计算 Best
9. 输出三个结果文件


============================================================
计算方式
============================================================

Precision-MMS：

每个 pred 在 gt 中找最高相似度：

    P = mean(
        max(sim(pred, gt))
    )


Recall-MMS：

每个 gt 在 pred 中找最高相似度：

    R = mean(
        max(sim(gt, pred))
    )


F1-like：

    F1 = 2PR / (P + R)


============================================================
Mean ± Std
============================================================

三个 Run 独立计算：

    Run 1 = mean(score_run1)
    Run 2 = mean(score_run2)
    Run 3 = mean(score_run3)

Mean：

    Mean = (Run1 + Run2 + Run3) / 3

Std：

    population std(Run1, Run2, Run3)


============================================================
Best
============================================================

每个 idx 三次结果取最高：

    Best_i = max(score_i1, score_i2, score_i3)

然后：

    Best = mean(Best_i)


============================================================
"""

import os
import glob
import json
import math
import re

import torch
from transformers import AutoTokenizer, AutoModel


# ============================================================
# 配置
# ============================================================

BASE_DIR = os.path.dirname(
    os.path.abspath(__file__)
)

EMBEDDING_MODEL_PATH = r"/nvme1/benchmark/QDA_Benchmark/scripts/model/all-MiniLM-L6-v2"


# ============================================================
# Task 2 Level
# ============================================================

LEVELS = [
    "code",
    "subtheme",
    "theme"
]


# ============================================================
# 模型名称映射
# ============================================================

def display_model_name(
    folder_name,
    filename
):

    # --------------------------------------------------------
    # 优先从文件名提取模型名称
    #
    # 例如：
    #
    # result_code_gpt-5_n3.jsonl
    # result_code_qwen3.7-max_n3.jsonl
    #
    # --------------------------------------------------------

    name = os.path.basename(
        filename
    )

    name = re.sub(
        r"\.jsonl$",
        "",
        name
    )

    # result_code_xxx_n3
    # result_subtheme_xxx_n3
    # result_theme_xxx_n3

    name = re.sub(
        r"^result_(code|subtheme|theme)_",
        "",
        name
    )

    name = re.sub(
        r"_n\d+$",
        "",
        name
    )

    # --------------------------------------------------------
    # 如果模型名里面有路径，例如：
    #
    # result_code_-ssd-amax-qda_models-Qwen2.5-72B-Instruct_n3
    #
    # 提取最后的模型名称
    # --------------------------------------------------------

    if "-qda_models-" in name:

        name = name.split(
            "-qda_models-"
        )[-1]

    # --------------------------------------------------------
    # 常见名称统一
    # --------------------------------------------------------

    mapping = {

        "gpt5":
            "gpt-5",

        "gpt-5":
            "gpt-5",

        "qwen37_max":
            "qwen3.7-max",

        "qwen3.7-max":
            "qwen3.7-max",

        "qwen25_7b":
            "qwen2.5-7b-Instruct",

        "qwen25_14b":
            "qwen2.5-14b-Instruct",

        "qwen25_32b":
            "qwen2.5-32b-Instruct",

        "qwen25_72b":
            "qwen2.5-72b-Instruct",

        "Qwen2.5-7B-Instruct":
            "qwen2.5-7b-Instruct",

        "Qwen2.5-14B-Instruct":
            "qwen2.5-14b-Instruct",

        "Qwen2.5-32B-Instruct":
            "qwen2.5-32b-Instruct",

        "Qwen2.5-72B-Instruct":
            "qwen2.5-72b-Instruct",

        "llama31_8b":
            "llama-3.1-8B-Instruct",

        "llama31_70b":
            "llama-3.1-70B-Instruct",

        "Llama-3.1-8B-Instruct":
            "llama-3.1-8B-Instruct",

        "Llama-3.1-70B-Instruct":
            "llama-3.1-70B-Instruct",

        "dsv4p":
            "deepseek-v4-pro",

        "deepseek-v4-pro":
            "deepseek-v4-pro"
    }

    return mapping.get(
        name,
        name
    )


# ============================================================
# Embedding
# ============================================================

def mean_pooling(
    model_output,
    attention_mask
):

    token_embeddings = model_output[0]

    input_mask_expanded = (
        attention_mask
        .unsqueeze(-1)
        .expand(
            token_embeddings.size()
        )
        .float()
    )

    return (
        torch.sum(
            token_embeddings
            * input_mask_expanded,
            dim=1
        )
        /
        torch.clamp(
            input_mask_expanded.sum(
                dim=1
            ),
            min=1e-9
        )
    )


def encode_texts(
    texts,
    tokenizer,
    model
):

    encoded = tokenizer(
        texts,
        padding=True,
        truncation=True,
        return_tensors="pt"
    )

    with torch.no_grad():

        output = model(
            **encoded
        )

    emb = mean_pooling(
        output,
        encoded["attention_mask"]
    )

    return torch.nn.functional.normalize(
        emb,
        p=2,
        dim=1
    )


# ============================================================
# Precision-MMS
# ============================================================

def precision_mms(
    pred_list,
    gt_list,
    tokenizer,
    model
):

    if (
        not pred_list
        or not gt_list
    ):

        return 0.0

    emb_pred = encode_texts(
        pred_list,
        tokenizer,
        model
    )

    emb_gt = encode_texts(
        gt_list,
        tokenizer,
        model
    )

    sim_matrix = torch.mm(
        emb_pred,
        emb_gt.T
    )

    max_sim = sim_matrix.max(
        dim=1
    ).values

    return max_sim.mean().item()


# ============================================================
# Recall-MMS
# ============================================================

def recall_mms(
    pred_list,
    gt_list,
    tokenizer,
    model
):

    if (
        not pred_list
        or not gt_list
    ):

        return 0.0

    emb_pred = encode_texts(
        pred_list,
        tokenizer,
        model
    )

    emb_gt = encode_texts(
        gt_list,
        tokenizer,
        model
    )

    sim_matrix = torch.mm(
        emb_gt,
        emb_pred.T
    )

    max_sim = sim_matrix.max(
        dim=1
    ).values

    return max_sim.mean().item()


# ============================================================
# F1-like
# ============================================================

def calculate_f1(
    precision,
    recall
):

    if (
        precision + recall
        <= 0
    ):

        return 0.0

    return (
        2
        * precision
        * recall
        /
        (
            precision
            + recall
        )
    )


# ============================================================
# 读取 JSONL
# ============================================================

def load_jsonl(
    input_file
):

    records = []

    with open(
        input_file,
        "r",
        encoding="utf-8"
    ) as f:

        for line_number, line in enumerate(
            f,
            1
        ):

            line = line.strip()

            if not line:
                continue

            try:

                records.append(
                    json.loads(line)
                )

            except json.JSONDecodeError as e:

                print(
                    f"[WARNING] "
                    f"{os.path.basename(input_file)} "
                    f"line {line_number}: "
                    f"{e}"
                )

    return records


# ============================================================
# 获取三次 Run
# ============================================================

def get_runs(
    record
):

    # --------------------------------------------------------
    # 第一种：
    #
    # "all_runs": [...]
    # --------------------------------------------------------

    if "all_runs" in record:

        runs = record[
            "all_runs"
        ]

        if isinstance(
            runs,
            list
        ):

            return runs[:3]


    # --------------------------------------------------------
    # 第二种：
    #
    # "runs": [...]
    # --------------------------------------------------------

    if "runs" in record:

        runs = record[
            "runs"
        ]

        if isinstance(
            runs,
            list
        ):

            return runs[:3]


        # ----------------------------------------------------
        # 第三种：
        #
        # "runs": {
        #     "run_1_initial": {...},
        #     ...
        # }
        # ----------------------------------------------------

        if isinstance(
            runs,
            dict
        ):

            return [

                runs.get(
                    "run_1_initial"
                ),

                runs.get(
                    "run_2_repeat_1"
                ),

                runs.get(
                    "run_3_repeat_2"
                )
            ]


    return [
        None,
        None,
        None
    ]


# ============================================================
# 获取 pred
# ============================================================

def get_pred(
    run
):

    if not isinstance(
        run,
        dict
    ):

        return None

    pred = run.get(
        "pred"
    )

    if pred is None:

        pred = run.get(
            "best_pred"
        )

    if not isinstance(
        pred,
        list
    ):

        return None

    return pred


# ============================================================
# 计算单个文件
# ============================================================

def calculate_file(
    input_file,
    tokenizer,
    model
):

    records = load_jsonl(
        input_file
    )

    precision_scores = []
    recall_scores = []
    f1_scores = []

    skipped = 0

    # ========================================================
    # 每个 idx
    # ========================================================

    for record in records:

        gt = record.get(
            "gt",
            []
        )

        # ----------------------------------------------------
        # gt 为空：跳过
        # ----------------------------------------------------

        if not gt:

            skipped += 1

            continue


        runs = get_runs(
            record
        )

        if (
            len(runs) != 3
            or any(
                run is None
                for run in runs
            )
        ):

            skipped += 1

            continue


        idx_precision = []
        idx_recall = []
        idx_f1 = []


        # ====================================================
        # 三次 Run
        # ====================================================

        for run in runs:

            pred = get_pred(
                run
            )

            # ------------------------------------------------
            # 空 pred：
            # 分数 = 0
            # ------------------------------------------------

            if not pred:

                p = 0.0
                r = 0.0
                f1 = 0.0

            else:

                p = precision_mms(
                    pred,
                    gt,
                    tokenizer,
                    model
                )

                r = recall_mms(
                    pred,
                    gt,
                    tokenizer,
                    model
                )

                f1 = calculate_f1(
                    p,
                    r
                )


            idx_precision.append(
                p
            )

            idx_recall.append(
                r
            )

            idx_f1.append(
                f1
            )


        precision_scores.append(
            idx_precision
        )

        recall_scores.append(
            idx_recall
        )

        f1_scores.append(
            idx_f1
        )


    total = len(
        precision_scores
    )

    if total == 0:

        return None


    # ========================================================
    # 三次 Run 分别求平均
    # ========================================================

    def run_means(
        scores
    ):

        return [

            sum(
                x[i]
                for x in scores
            ) / total

            for i in range(3)
        ]


    p_runs = run_means(
        precision_scores
    )

    r_runs = run_means(
        recall_scores
    )

    f_runs = run_means(
        f1_scores
    )


    # ========================================================
    # Mean
    # ========================================================

    def mean3(
        values
    ):

        return sum(values) / 3


    p_mean = mean3(
        p_runs
    )

    r_mean = mean3(
        r_runs
    )

    f_mean = mean3(
        f_runs
    )


    # ========================================================
    # Std
    # ========================================================

    def std3(
        values,
        mean
    ):

        return math.sqrt(
            sum(
                (
                    x - mean
                ) ** 2
                for x in values
            ) / 3
        )


    p_std = std3(
        p_runs,
        p_mean
    )

    r_std = std3(
        r_runs,
        r_mean
    )

    f_std = std3(
        f_runs,
        f_mean
    )


    # ========================================================
    # Best
    #
    # 每个 idx 三次取最高
    # ========================================================

    p_best = sum(
        max(x)
        for x in precision_scores
    ) / total

    r_best = sum(
        max(x)
        for x in recall_scores
    ) / total

    f_best = sum(
        max(x)
        for x in f1_scores
    ) / total


    return {

        "entries":
            total,

        "skipped":
            skipped,

        # Precision
        "precision_mean":
            p_mean,

        "precision_std":
            p_std,

        "precision_best":
            p_best,

        # Recall
        "recall_mean":
            r_mean,

        "recall_std":
            r_std,

        "recall_best":
            r_best,

        # F1
        "f1_mean":
            f_mean,

        "f1_std":
            f_std,

        "f1_best":
            f_best
    }


# ============================================================
# 找模型文件夹
# ============================================================

model_dirs = [

    path

    for path in sorted(
        glob.glob(
            os.path.join(
                BASE_DIR,
                "*"
            )
        )
    )

    if os.path.isdir(
        path
    )

    and not os.path.basename(
        path
    ).startswith(".")
]


if not model_dirs:

    raise FileNotFoundError(
        "没有找到模型文件夹"
    )


# ============================================================
# 加载 Embedding 模型
# ============================================================

print(
    "=" * 80
)

print(
    "Loading embedding model..."
)

print(
    EMBEDDING_MODEL_PATH
)

print(
    "=" * 80
)

tokenizer = AutoTokenizer.from_pretrained(
    EMBEDDING_MODEL_PATH
)

embedding_model = AutoModel.from_pretrained(
    EMBEDDING_MODEL_PATH
)

embedding_model.eval()


# ============================================================
# results
#
# results[model][level]
# ============================================================

results = {}


# ============================================================
# 处理所有模型
# ============================================================

for model_dir in model_dirs:

    folder_name = os.path.basename(
        model_dir
    )

    print()
    print(
        "=" * 80
    )

    print(
        f"Processing folder: "
        f"{folder_name}"
    )

    print(
        "=" * 80
    )


    model_results = {}


    # ========================================================
    # 找 code / subtheme / theme
    # ========================================================

    for level in LEVELS:

        pattern = os.path.join(
            model_dir,
            f"result_{level}_*_n*.jsonl"
        )

        files = sorted(
            glob.glob(
                pattern
            )
        )

        if not files:

            print(
                f"[WARNING] "
                f"Missing {level} file"
            )

            continue


        # ----------------------------------------------------
        # 正常情况下每个 level 只有一个文件
        # ----------------------------------------------------

        input_file = files[0]


        model_name = display_model_name(
            folder_name,
            input_file
        )


        print()
        print(
            f"[{level.upper()}]"
        )

        print(
            f"File: "
            f"{os.path.basename(input_file)}"
        )

        result = calculate_file(
            input_file,
            tokenizer,
            embedding_model
        )


        if result is None:

            print(
                "[WARNING] "
                "No valid data."
            )

            continue


        model_results[
            level
        ] = result


        print(
            f"Precision: "
            f"{result['precision_mean'] * 100:.2f} "
            f"± "
            f"{result['precision_std'] * 100:.2f} "
            f"| Best "
            f"{result['precision_best'] * 100:.2f}"
        )

        print(
            f"Recall:    "
            f"{result['recall_mean'] * 100:.2f} "
            f"± "
            f"{result['recall_std'] * 100:.2f} "
            f"| Best "
            f"{result['recall_best'] * 100:.2f}"
        )

        print(
            f"F1-like:   "
            f"{result['f1_mean'] * 100:.2f} "
            f"± "
            f"{result['f1_std'] * 100:.2f} "
            f"| Best "
            f"{result['f1_best'] * 100:.2f}"
        )


    if model_results:

        results[
            model_name
        ] = model_results


# ============================================================
# 生成 Markdown 表格
# ============================================================

def generate_table(
    metric,
    best=False
):

    if best:

        title = (
            f"{metric} - Best (%)"
        )

    else:

        title = (
            f"{metric} - Mean ± Std (%)"
        )


    lines = []

    lines.append(
        f"## {title}"
    )

    lines.append("")

    lines.append(
        "|Model|Code|Sub-Theme|Theme|"
    )

    lines.append(
        "|---|---|---|---|"
    )


    for model in results:

        row = [
            model
        ]


        for level in LEVELS:

            if (
                level not in
                results[model]
            ):

                row.append(
                    "-"
                )

                continue


            result = results[
                model
            ][
                level
            ]


            if metric == "Precision":

                mean_key = (
                    "precision_mean"
                )

                std_key = (
                    "precision_std"
                )

                best_key = (
                    "precision_best"
                )


            elif metric == "Recall":

                mean_key = (
                    "recall_mean"
                )

                std_key = (
                    "recall_std"
                )

                best_key = (
                    "recall_best"
                )


            else:

                mean_key = (
                    "f1_mean"
                )

                std_key = (
                    "f1_std"
                )

                best_key = (
                    "f1_best"
                )


            if best:

                value = (
                    f"{result[best_key] * 100:.2f}"
                )

            else:

                value = (
                    f"{result[mean_key] * 100:.2f}"
                    f" ± "
                    f"{result[std_key] * 100:.2f}"
                )


            row.append(
                value
            )


        lines.append(
            "|"
            + "|".join(row)
            + "|"
        )


    return "\n".join(
        lines
    )


# ============================================================
# 生成全部表格
# ============================================================

tables = {}


for metric in [
    "Precision",
    "Recall",
    "F1-like"
]:

    tables[
        f"{metric}_mean"
    ] = generate_table(
        metric,
        best=False
    )

    tables[
        f"{metric}_best"
    ] = generate_table(
        metric,
        best=True
    )


# ============================================================
# 输出文件
# ============================================================

output_files = {

    "Precision": (
        "precision_mms_results.txt"
    ),

    "Recall": (
        "recall_mms_results.txt"
    ),

    "F1-like": (
        "f1_like_results.txt"
    )
}


for metric, filename in output_files.items():

    output_path = os.path.join(
        BASE_DIR,
        filename
    )


    mean_table = tables[
        f"{metric}_mean"
    ]

    best_table = tables[
        f"{metric}_best"
    ]


    with open(
        output_path,
        "w",
        encoding="utf-8"
    ) as f:

        f.write(
            mean_table
        )

        f.write(
            "\n\n"
        )

        f.write(
            best_table
        )


    print()
    print(
        "=" * 80
    )

    print(
        mean_table
    )

    print()

    print(
        best_table
    )

    print(
        f"\nSaved to: "
        f"{output_path}"
    )


print()
print(
    "=" * 80
)

print(
    "ALL FINISHED!"
)

print(
    "=" * 80
)