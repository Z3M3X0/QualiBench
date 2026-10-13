import json
import glob
import os
import math
import re


# ============================================================
# 配置
# ============================================================

BASE_DIR = os.path.dirname(
    os.path.abspath(__file__)
)

OUTPUT_FILE = os.path.join(
    BASE_DIR,
    "task2_results.txt"
)


LEVELS = [
    "code",
    "subtheme",
    "theme"
]


# ============================================================
# 模型名称映射
# ============================================================

def display_model_name(model_name):

    mapping = {

        # ----------------------------------------------------
        # Close source
        # ----------------------------------------------------

        "dsv4p":
            "deepseek-v4-pro",

        "deepseek-v4-pro":
            "deepseek-v4-pro",

        "gpt5":
            "gpt-5",

        "gpt-5":
            "gpt-5",

        "qwen37_max":
            "qwen3.7-max",

        "qwen3.7-max":
            "qwen3.7-max",


        # ----------------------------------------------------
        # Qwen
        # ----------------------------------------------------

        "qwen25_7b":
            "qwen2.5-7b-Instruct",

        "qwen25_14b":
            "qwen2.5-14b-Instruct",

        "qwen25_32b":
            "qwen2.5-32b-Instruct",

        "qwen25_72b":
            "qwen2.5-72b-Instruct",


        # ----------------------------------------------------
        # Llama
        # ----------------------------------------------------

        "llama31_8b":
            "llama-3.1-8B-Instruct",

        "llama31_70b":
            "llama-3.1-70B-Instruct"
    }

    return mapping.get(
        model_name,
        model_name
    )


# ============================================================
# 读取 JSONL
# ============================================================

def load_jsonl(input_file):

    data = []

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

                data.append(
                    json.loads(line)
                )

            except json.JSONDecodeError as e:

                print(
                    f"[WARNING] "
                    f"{os.path.basename(input_file)} "
                    f"line {line_number}: {e}"
                )

    return data


# ============================================================
# 从文件名获取 level
#
# 例如：
#
# result_code_gpt-5_n3.jsonl
# result_subtheme_gpt-5_n3.jsonl
# result_theme_gpt-5_n3.jsonl
# ============================================================

def get_level_from_filename(filename):

    filename_lower = filename.lower()

    for level in LEVELS:

        if (
            f"result_{level}_"
            in filename_lower
        ):

            return level

    return None


# ============================================================
# 从文件名获取模型名称
#
# 例如：
#
# result_code_gpt-5_n3.jsonl
#
# → gpt-5
#
# result_code_-ssd-amax-qda_models-Qwen2.5-72B-Instruct_n3.jsonl
#
# → qwen2.5-72b-Instruct
# ============================================================

def get_model_from_filename(filename):

    name = os.path.basename(
        filename
    )

    # 去掉 .jsonl

    name = re.sub(
        r"\.jsonl$",
        "",
        name,
        flags=re.IGNORECASE
    )

    # 去掉 result_code_ / result_theme_ / result_subtheme_

    name = re.sub(
        r"^result_(code|subtheme|theme)_",
        "",
        name,
        flags=re.IGNORECASE
    )

    # 去掉最后的 _n3

    name = re.sub(
        r"_n\d+$",
        "",
        name,
        flags=re.IGNORECASE
    )

    # --------------------------------------------------------
    # 处理类似：
    #
    # -ssd-amax-qda_models-Qwen2.5-72B-Instruct
    # --------------------------------------------------------

    if "Qwen2.5-72B-Instruct" in name:

        name = "qwen25_72b"

    elif "Qwen2.5-32B-Instruct" in name:

        name = "qwen25_32b"

    elif "Qwen2.5-14B-Instruct" in name:

        name = "qwen25_14b"

    elif "Qwen2.5-7B-Instruct" in name:

        name = "qwen25_7b"

    elif "Llama-3.1-70B-Instruct" in name:

        name = "llama31_70b"

    elif "Llama-3.1-8B-Instruct" in name:

        name = "llama31_8b"

    elif name.lower() == "gpt-5":

        name = "gpt5"

    elif name.lower() == "qwen3.7-max":

        name = "qwen37_max"

    elif name.lower() == "deepseek-v4-pro":

        name = "dsv4p"

    elif name.lower() == "dsv4p":

        name = "dsv4p"

    return display_model_name(
        name
    )


# ============================================================
# 获取三个 run
#
# Task 2 标准格式：
#
# "all_runs": [
#     {
#         "pred": [...],
#         "recall": 0.2225
#     },
#     {
#         "pred": [...],
#         "recall": 0.227
#     },
#     {
#         "pred": [...],
#         "recall": 0.268
#     }
# ]
#
# 也兼容：
#
# "runs": [...]
#
# 或：
#
# "runs": {
#     "run_1_initial": {...},
#     ...
# }
# ============================================================

def get_runs(item):

    # ========================================================
    # all_runs
    # ========================================================

    if "all_runs" in item:

        runs = item["all_runs"]

        if isinstance(
            runs,
            list
        ):

            return runs[:3]


    # ========================================================
    # runs
    # ========================================================

    if "runs" in item:

        runs = item["runs"]

        # --------------------------------------------
        # dict
        # --------------------------------------------

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

        # --------------------------------------------
        # list
        # --------------------------------------------

        if isinstance(
            runs,
            list
        ):

            return runs[:3]


    return [
        None,
        None,
        None
    ]


# ============================================================
# 获取单个 run 的 score
#
# Task 2 可能使用：
#
# recall
#
# 或：
#
# mms
# ============================================================

def get_score(run):

    if not isinstance(
        run,
        dict
    ):

        return None


    # ========================================================
    # Recall
    # ========================================================

    if "recall" in run:

        try:

            return float(
                run["recall"]
            )

        except (
            TypeError,
            ValueError
        ):

            return None


    # ========================================================
    # MMS
    # ========================================================

    if "mms" in run:

        try:

            return float(
                run["mms"]
            )

        except (
            TypeError,
            ValueError
        ):

            return None


    # ========================================================
    # 某些文件可能直接叫 similarity
    # 虽然 Task 2 正常不需要，但兼容一下
    # ========================================================

    if "similarity" in run:

        try:

            return float(
                run["similarity"]
            )

        except (
            TypeError,
            ValueError
        ):

            return None


    return None


# ============================================================
# 计算单个 Task 2 文件
#
# Mean：
#
#   Run1 所有 idx 的平均
#   Run2 所有 idx 的平均
#   Run3 所有 idx 的平均
#
#   最后：
#
#   Mean = (Run1 + Run2 + Run3) / 3
#
#
# Std：
#
#   三个 Run-level mean 的总体标准差
#
#
# Best：
#
#   每个 idx：
#
#   max(Run1, Run2, Run3)
#
#   然后所有 idx 的 best score 求平均
# ============================================================

def calculate_file(input_file):

    data = load_jsonl(
        input_file
    )

    # ========================================================
    # 保存每个 idx 的三个 score
    #
    # [
    #     [0.22, 0.23, 0.27],
    #     [0.31, 0.29, 0.35],
    #     ...
    # ]
    # ========================================================

    valid_items = []

    for item in data:

        runs = get_runs(
            item
        )

        if len(runs) != 3:

            continue


        scores = []

        valid = True


        for run in runs:

            score = get_score(
                run
            )

            if score is None:

                valid = False

                break

            scores.append(
                score
            )


        if valid:

            valid_items.append(
                scores
            )


    # ========================================================
    # 数据检查
    # ========================================================

    N = len(
        valid_items
    )

    if N == 0:

        return None


    # ========================================================
    # 三个 Run 分别求平均
    # ========================================================

    run_means = []


    for run_index in range(3):

        total = sum(
            scores[run_index]
            for scores in valid_items
        )

        run_mean = (
            total / N
        )

        run_means.append(
            run_mean
        )


    # ========================================================
    # Mean
    # ========================================================

    mean = (
        sum(run_means)
        / 3
    )


    # ========================================================
    # Std
    #
    # 三个 Run-level mean 的总体标准差
    # ========================================================

    variance = sum(
        (
            x - mean
        ) ** 2
        for x in run_means
    ) / 3

    std = math.sqrt(
        variance
    )


    # ========================================================
    # Best
    #
    # 每个 idx 的三个 score 取最高
    # 然后再求整体平均
    # ========================================================

    best_scores = [

        max(scores)

        for scores
        in valid_items
    ]


    best = (
        sum(best_scores)
        / N
    )


    # ========================================================
    # 返回
    # ========================================================

    return {

        "entries":
            N,

        "run1":
            run_means[0],

        "run2":
            run_means[1],

        "run3":
            run_means[2],

        "mean":
            mean,

        "std":
            std,

        "best":
            best
    }


# ============================================================
# 找 Task 2 文件
#
# 当前目录结构：
#
# task2_result/
#
#     task2_gpt5/
#         result_code_gpt-5_n3.jsonl
#         result_subtheme_gpt-5_n3.jsonl
#         result_theme_gpt-5_n3.jsonl
#
#     task2_qwen37_max/
#         ...
#
#     task2_qwen25_72b/
#         ...
#
# ============================================================

all_files = sorted(
    glob.glob(
        os.path.join(
            BASE_DIR,
            "*",
            "*.jsonl"
        )
    )
)


if not all_files:

    raise FileNotFoundError(
        "没有找到 Task 2 JSONL 文件"
    )


# ============================================================
# results[model][level]
# ============================================================

results = {}


# ============================================================
# 处理所有文件
# ============================================================

for input_file in all_files:

    filename = os.path.basename(
        input_file
    )

    level = get_level_from_filename(
        filename
    )

    if level is None:

        print(
            f"[WARNING] "
            f"无法识别 level: "
            f"{filename}"
        )

        continue


    model_name = get_model_from_filename(
        filename
    )


    if model_name not in results:

        results[
            model_name
        ] = {}


    print()
    print(
        "=" * 80
    )

    print(
        f"Processing: {filename}"
    )

    print(
        f"Model: {model_name}"
    )

    print(
        f"Level: {level}"
    )

    print(
        "=" * 80
    )


    result = calculate_file(
        input_file
    )


    if result is None:

        print(
            "[WARNING] "
            "No valid data"
        )

        continue


    results[
        model_name
    ][
        level
    ] = result


    print(
        f"N     = {result['entries']}"
    )

    print(
        f"Run 1 = "
        f"{result['run1'] * 100:.2f}%"
    )

    print(
        f"Run 2 = "
        f"{result['run2'] * 100:.2f}%"
    )

    print(
        f"Run 3 = "
        f"{result['run3'] * 100:.2f}%"
    )

    print(
        f"Mean  = "
        f"{result['mean'] * 100:.2f}%"
    )

    print(
        f"Std   = "
        f"{result['std'] * 100:.2f}%"
    )

    print(
        f"Best  = "
        f"{result['best'] * 100:.2f}%"
    )


# ============================================================
# 模型排序
#
# 按照你之前表格的顺序
# ============================================================

MODEL_ORDER = [

    "deepseek-v4-pro",

    "gpt-5",

    "qwen3.7-max",

    "qwen2.5-7b-Instruct",

    "qwen2.5-14b-Instruct",

    "qwen2.5-32b-Instruct",

    "qwen2.5-72b-Instruct",

    "llama-3.1-8B-Instruct",

    "llama-3.1-70B-Instruct"
]


def sorted_models():

    existing = set(
        results.keys()
    )

    ordered = []

    # 先按照指定顺序
    for model in MODEL_ORDER:

        if model in existing:

            ordered.append(
                model
            )

    # 再放其他模型
    for model in results:

        if model not in ordered:

            ordered.append(
                model
            )

    return ordered


# ============================================================
# 生成 Mean ± Std 表
# ============================================================

def generate_mean_table():

    lines = []

    lines.append(
        "## Mean ± Std (%)"
    )

    lines.append("")

    lines.append(
        "|Model|Code|Sub-Theme|Theme|"
    )

    lines.append(
        "|---|---|---|---|"
    )


    for model in sorted_models():

        row = [
            model
        ]


        for level in LEVELS:

            if (
                level
                not in results[model]
            ):

                row.append("-")

                continue


            result = results[
                model
            ][
                level
            ]


            value = (
                f"{result['mean'] * 100:.2f}"
                f" ± "
                f"{result['std'] * 100:.2f}"
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
# 生成 Best 表
# ============================================================

def generate_best_table():

    lines = []

    lines.append(
        "## Best (%)"
    )

    lines.append("")

    lines.append(
        "|Model|Code|Sub-Theme|Theme|"
    )

    lines.append(
        "|---|---|---|---|"
    )


    for model in sorted_models():

        row = [
            model
        ]


        for level in LEVELS:

            if (
                level
                not in results[model]
            ):

                row.append("-")

                continue


            result = results[
                model
            ][
                level
            ]


            value = (
                f"{result['best'] * 100:.2f}"
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
# 输出
# ============================================================

mean_table = generate_mean_table()

best_table = generate_best_table()


with open(
    OUTPUT_FILE,
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


# ============================================================
# 控制台输出
# ============================================================

print()
print(
    "=" * 80
)

print(
    mean_table
)

print()
print(
    "=" * 80
)

print(
    best_table
)

print()
print(
    "=" * 80
)

print(
    "Finished!"
)

print(
    f"Results saved to: "
    f"{OUTPUT_FILE}"
)

print(
    "=" * 80
)