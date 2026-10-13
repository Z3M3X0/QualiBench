import json
import glob
import os
import math


# ============================================================
# 配置
# ============================================================

BASE_DIR = os.path.dirname(
    os.path.abspath(__file__)
)

OUTPUT_FILE = os.path.join(
    BASE_DIR,
    "task1_results.txt"
)

DOMAINS = [
    "Technology",
    "Education",
    "Medicine",
    "Finance",
    "Law",
    "Art",
    "Sustainability"
]

THRESHOLD = 0.7


# ============================================================
# 模型名称
# ============================================================

def display_model_name(model_name):

    mapping = {

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

        "qwen25_7b":
            "qwen2.5-7b-Instruct",

        "qwen25_14b":
            "qwen2.5-14b-Instruct",

        "qwen25_32b":
            "qwen2.5-32b-Instruct",

        "qwen25_72b":
            "qwen2.5-72b-Instruct",

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
# 获取三个 run
# ============================================================

def get_runs(item):

    # ========================================================
    # 格式 1：
    #
    # "runs": {
    #     "run_1_initial": {...},
    #     "run_2_repeat_1": {...},
    #     "run_3_repeat_2": {...}
    # }
    # ========================================================

    if "runs" in item:

        runs = item["runs"]

        if isinstance(runs, dict):

            return [
                runs.get("run_1_initial"),
                runs.get("run_2_repeat_1"),
                runs.get("run_3_repeat_2")
            ]

        # 某些文件可能是：
        #
        # "runs": [
        #     {...},
        #     {...},
        #     {...}
        # ]

        if isinstance(runs, list):

            return runs[:3]


    # ========================================================
    # 格式 2：
    #
    # "all_runs": [
    #     {...},
    #     {...},
    #     {...}
    # ]
    # ========================================================

    if "all_runs" in item:

        runs = item["all_runs"]

        if isinstance(runs, list):

            return runs[:3]


    # ========================================================
    # 无法识别
    # ========================================================

    return [
        None,
        None,
        None
    ]


# ============================================================
# 计算单个 domain
#
# Task 1 统计逻辑
#
# Run 1：
#   独立计算 similarity > 0.7 的 Pass Rate
#
# Run 2：
#   独立计算 similarity > 0.7 的 Pass Rate
#
# Run 3：
#   独立计算 similarity > 0.7 的 Pass Rate
#
# Mean：
#   (Run1 Pass Rate + Run2 Pass Rate + Run3 Pass Rate) / 3
#
# Std：
#   三个独立 Pass Rate 的总体标准差
#
# Best：
#   每个 idx 的三次 similarity 中，
#   只要有一次 > 0.7，该 idx 就算通过
# ============================================================

def calculate_domain(input_file):

    data = load_jsonl(input_file)

    # ========================================================
    # 判断文件是哪种形式
    #
    # 1. 三次运行：
    #    runs / all_runs
    #
    # 2. 单次运行：
    #    similarity
    # ========================================================

    three_run_items = []
    single_run_items = []

    for item in data:

        # ----------------------------------------------------
        # 优先检查三次运行
        # ----------------------------------------------------

        runs = get_runs(item)

        if (
            len(runs) == 3
            and all(
                isinstance(run, dict)
                and "similarity" in run
                for run in runs
            )
        ):

            try:

                similarities = [
                    float(run["similarity"])
                    for run in runs
                ]

                three_run_items.append(
                    similarities
                )

                continue

            except (
                TypeError,
                ValueError
            ):

                pass

        # ----------------------------------------------------
        # 检查单次运行
        # ----------------------------------------------------

        if "similarity" in item:

            try:

                similarity = float(
                    item["similarity"]
                )

                single_run_items.append(
                    similarity
                )

            except (
                TypeError,
                ValueError
            ):

                continue

    # ========================================================
    # 如果同时出现单次和三次格式
    # ========================================================

    if (
        len(three_run_items) > 0
        and len(single_run_items) > 0
    ):

        print(
            f"[WARNING] {os.path.basename(input_file)} "
            f"同时包含三次运行和单次运行数据，"
            f"将只使用三次运行数据。"
        )

    # ========================================================
    # 情况 A：三次运行
    # ========================================================

    if len(three_run_items) > 0:

        valid_items = three_run_items

        N = len(valid_items)

        # ----------------------------------------------------
        # 三个 Run 独立计算 Pass Rate
        # ----------------------------------------------------

        run_pass_rates = []

        for run_index in range(3):

            passed = sum(
                1
                for similarities in valid_items
                if similarities[run_index] > THRESHOLD
            )

            pass_rate = passed / N

            run_pass_rates.append(
                pass_rate
            )

        # ----------------------------------------------------
        # Mean
        # ----------------------------------------------------

        mean = (
            sum(run_pass_rates) / 3
        )

        # ----------------------------------------------------
        # Std
        # ----------------------------------------------------

        variance = sum(
            (x - mean) ** 2
            for x in run_pass_rates
        ) / 3

        std = math.sqrt(
            variance
        )

        # ----------------------------------------------------
        # Best
        #
        # 每个 idx 三次只要一次通过
        # ----------------------------------------------------

        best_passed = sum(
            1
            for similarities in valid_items
            if max(similarities) > THRESHOLD
        )

        best = (
            best_passed / N
        )

        return {

            "entries": N,

            "n_runs": 3,

            "run1":
                run_pass_rates[0],

            "run2":
                run_pass_rates[1],

            "run3":
                run_pass_rates[2],

            "mean":
                mean,

            "std":
                std,

            "best":
                best
        }

    # ========================================================
    # 情况 B：只有一次运行
    # ========================================================

    elif len(single_run_items) > 0:

        N = len(
            single_run_items
        )

        passed = sum(
            1
            for similarity in single_run_items
            if similarity > THRESHOLD
        )

        pass_rate = (
            passed / N
        )

        # 单次运行没有 Run2 / Run3
        #
        # Mean = 单次 Pass Rate
        # Std  = 0
        #
        # Best = 不再区分 best，
        # 因为只有一次结果，所以等于单次 Pass Rate

        return {

            "entries": N,

            "n_runs": 1,

            "run1":
                pass_rate,

            "run2":
                None,

            "run3":
                None,

            "mean":
                pass_rate,

            "std":
                0.0,

            "best":
                pass_rate
        }

    # ========================================================
    # 没有有效数据
    # ========================================================

    return None
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
    if os.path.isdir(path)
]


if not model_dirs:

    raise FileNotFoundError(
        "没有找到模型文件夹"
    )


# ============================================================
# results[model][domain]
# ============================================================

results = {}


# ============================================================
# 处理
# ============================================================

for model_dir in model_dirs:

    model_folder = os.path.basename(
        model_dir
    )

    model_name = display_model_name(
        model_folder
    )

    results[
        model_name
    ] = {}

    print()
    print("=" * 80)
    print(
        f"Processing: {model_name}"
    )
    print("=" * 80)

    for domain in DOMAINS:

        input_file = os.path.join(
            model_dir,
            f"{domain}.jsonl"
        )

        if not os.path.exists(
            input_file
        ):

            print(
                f"[WARNING] Missing: "
                f"{domain}.jsonl"
            )

            continue

        result = calculate_domain(
            input_file
        )

        if result is None:

            continue

        results[
            model_name
        ][
            domain
        ] = result

        if result["n_runs"] == 3:

            print(
                f"{domain}: "
                f"N={result['entries']} | "
                f"Run1={result['run1'] * 100:.2f}% | "
                f"Run2={result['run2'] * 100:.2f}% | "
                f"Run3={result['run3'] * 100:.2f}% | "
                f"Mean={result['mean'] * 100:.2f}% | "
                f"Std={result['std'] * 100:.2f}% | "
                f"Best={result['best'] * 100:.2f}%"
            )

        else:

            print(
                f"{domain}: "
                f"N={result['entries']} | "
                f"Single Run={result['run1'] * 100:.2f}% | "
                f"Mean={result['mean'] * 100:.2f}% | "
                f"Std={result['std'] * 100:.2f}% | "
                f"Best={result['best'] * 100:.2f}%"
            )


# ============================================================
# 生成表格
# ============================================================

def generate_table(
    metric
):

    if metric == "mean":

        title = "Mean ± Std (%)"

    else:

        title = "Best (%)"

    lines = []

    lines.append(
        f"## {title}"
    )

    lines.append("")

    lines.append(
        "|Model|"
        + "|".join(DOMAINS)
        + "|"
    )

    lines.append(
        "|---|"
        + "|".join(
            ["---"] * len(DOMAINS)
        )
        + "|"
    )

    for model in results:

        row = [
            model
        ]

        for domain in DOMAINS:

            if (
                domain not in results[model]
            ):

                row.append("-")

                continue

            result = results[
                model
            ][
                domain
            ]

            if metric == "mean":

                value = (
                    f"{result['mean']*100:.2f}"
                    f" ± "
                    f"{result['std']*100:.2f}"
                )

            else:

                value = (
                    f"{result['best']*100:.2f}"
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

mean_table = generate_table(
    "mean"
)

best_table = generate_table(
    "best"
)

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


print()
print("=" * 80)
print(mean_table)

print()
print("=" * 80)
print(best_table)

print()
print("=" * 80)
print(
    f"Finished!\n"
    f"Results saved to: {OUTPUT_FILE}"
)
print("=" * 80)