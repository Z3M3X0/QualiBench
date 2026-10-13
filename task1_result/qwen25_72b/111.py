import json
import glob
import os


# ============================================================
# 配置
# ============================================================

# 当前 Python 文件所在文件夹
INPUT_DIR = os.path.dirname(os.path.abspath(__file__))

# 输出文件，与 Python 文件放在同一个文件夹
OUTPUT_FILE = os.path.join(INPUT_DIR, "results.txt")


# ============================================================
# 自动识别 Runs 和 Metric
# ============================================================

def detect_runs(item):

    # Task 1:
    # "runs": {
    #     "run_1_initial": {...},
    #     "run_2_repeat_1": {...},
    #     "run_3_repeat_2": {...}
    # }

    if "runs" in item:
        return item["runs"], "runs"

    # Task 2:
    # "all_runs": [
    #     {...},
    #     {...},
    #     {...}
    # ]

    if "all_runs" in item:
        return item["all_runs"], "all_runs"

    return None, None


def detect_metric(runs):

    # --------------------------------------------------------
    # runs 是 dict
    # Task 1 格式
    # --------------------------------------------------------

    if isinstance(runs, dict):

        metric_candidates = [
            "similarity",
            "recall",
            "mms"
        ]

        for metric in metric_candidates:

            if all(
                isinstance(run, dict)
                and metric in run
                for run in runs.values()
            ):
                return metric

    # --------------------------------------------------------
    # runs 是 list
    # Task 2 格式
    # --------------------------------------------------------

    elif isinstance(runs, list):

        metric_candidates = [
            "similarity",
            "recall",
            "mms"
        ]

        for metric in metric_candidates:

            if all(
                isinstance(run, dict)
                and metric in run
                for run in runs
            ):
                return metric

    return None


def extract_values(runs, metric):

    # --------------------------------------------------------
    # dict
    # --------------------------------------------------------

    if isinstance(runs, dict):

        values = [
            run[metric]
            for run in runs.values()
            if isinstance(run, dict)
            and metric in run
        ]

    # --------------------------------------------------------
    # list
    # --------------------------------------------------------

    elif isinstance(runs, list):

        values = [
            run[metric]
            for run in runs
            if isinstance(run, dict)
            and metric in run
        ]

    else:

        values = []

    return values


# ============================================================
# 计算单个 JSONL 文件
# ============================================================

def calculate_file(input_file):

    data = []

    # --------------------------------------------------------
    # 读取 JSONL
    # --------------------------------------------------------

    with open(
        input_file,
        "r",
        encoding="utf-8"
    ) as f:

        for line_number, line in enumerate(f, 1):

            line = line.strip()

            if not line:
                continue

            try:

                item = json.loads(line)

                data.append(item)

            except json.JSONDecodeError as e:

                print(
                    f"[WARNING] {input_file} "
                    f"第 {line_number} 行解析失败: {e}"
                )

    # ========================================================
    # 每个 idx
    # ========================================================

    idx_means = []
    idx_bests = []

    metric_name = None
    run_container = None

    for item in data:

        idx = item.get("idx")

        # ----------------------------------------------------
        # 自动识别 runs / all_runs
        # ----------------------------------------------------

        runs, current_container = detect_runs(item)

        if runs is None:

            print(
                f"[WARNING] "
                f"{os.path.basename(input_file)} "
                f"idx={idx} "
                f"找不到 'runs' 或 'all_runs'，跳过"
            )

            continue

        # ----------------------------------------------------
        # 自动识别 metric
        # ----------------------------------------------------

        current_metric = detect_metric(runs)

        if current_metric is None:

            print(
                f"[WARNING] "
                f"{os.path.basename(input_file)} "
                f"idx={idx} "
                f"无法识别 metric "
                f"(similarity / recall / mms)，跳过"
            )

            continue

        # ----------------------------------------------------
        # 检查文件内部格式是否一致
        # ----------------------------------------------------

        if metric_name is None:

            metric_name = current_metric

        elif metric_name != current_metric:

            print(
                f"[WARNING] "
                f"{os.path.basename(input_file)} "
                f"idx={idx} "
                f"metric 不一致："
                f"{metric_name} -> {current_metric}，跳过"
            )

            continue

        if run_container is None:

            run_container = current_container

        elif run_container != current_container:

            print(
                f"[WARNING] "
                f"{os.path.basename(input_file)} "
                f"idx={idx} "
                f"runs 格式不一致："
                f"{run_container} -> {current_container}，跳过"
            )

            continue

        # ----------------------------------------------------
        # 提取三个 metric
        # ----------------------------------------------------

        values = extract_values(
            runs,
            current_metric
        )

        # ----------------------------------------------------
        # 必须正好有 3 个结果
        # ----------------------------------------------------

        if len(values) != 3:

            print(
                f"[WARNING] "
                f"{os.path.basename(input_file)} "
                f"idx={idx} "
                f"有 {len(values)} 个 "
                f"{current_metric}，跳过"
            )

            continue

        # ====================================================
        # 每个 idx
        # ====================================================

        # 三次结果平均
        idx_mean = sum(values) / 3

        # 三次结果中最好的一次
        idx_best = max(values)

        idx_means.append(idx_mean)
        idx_bests.append(idx_best)

    # ========================================================
    # 检查数据
    # ========================================================

    N = len(idx_means)

    if N == 0:

        return None

    # ========================================================
    # Mean
    # ========================================================

    overall_mean = sum(idx_means) / N

    # ========================================================
    # Variance
    # ========================================================

    variance = sum(
        (x - overall_mean) ** 2
        for x in idx_means
    ) / N

    # ========================================================
    # Standard Deviation
    # ========================================================

    std = variance ** 0.5

    # ========================================================
    # Best
    # ========================================================

    overall_best = sum(idx_bests) / N

    return {
        "file": os.path.basename(input_file),
        "entries": N,
        "container": run_container,
        "metric": metric_name,
        "mean": overall_mean,
        "variance": variance,
        "std": std,
        "best": overall_best
    }


# ============================================================
# 批量处理
# ============================================================

jsonl_files = sorted(
    glob.glob(
        os.path.join(
            INPUT_DIR,
            "*.jsonl"
        )
    )
)

if not jsonl_files:

    raise FileNotFoundError(
        f"在 {INPUT_DIR} 中没有找到 JSONL 文件"
    )


print("=" * 70)

print(
    f"Found {len(jsonl_files)} JSONL files"
)

print("=" * 70)


results = []


for input_file in jsonl_files:

    print(
        f"Processing: "
        f"{os.path.basename(input_file)}"
    )

    result = calculate_file(
        input_file
    )

    if result is None:

        print(
            "  No valid data found."
        )

        print()

        continue

    results.append(result)

    # ========================================================
    # 百分比
    # ========================================================

    mean_100 = result["mean"] * 100

    std_100 = result["std"] * 100

    best_100 = result["best"] * 100

    # ========================================================
    # 输出
    # ========================================================

    print(
        f"  Container        = "
        f"{result['container']}"
    )

    print(
        f"  Metric           = "
        f"{result['metric']}"
    )

    print(
        f"  N                = "
        f"{result['entries']}"
    )

    print(
        f"  Mean             = "
        f"{result['mean']:.6f}"
    )

    print(
        f"  Variance         = "
        f"{result['variance']:.6f}"
    )

    print(
        f"  Std              = "
        f"{result['std']:.6f}"
    )

    print(
        f"  Mean ± Std (%)   = "
        f"{mean_100:.2f} ± {std_100:.2f}"
    )

    print(
        f"  Best             = "
        f"{result['best']:.6f}"
    )

    print(
        f"  Best (%)         = "
        f"{best_100:.2f}"
    )

    print()


# ============================================================
# 写入 TXT
# ============================================================

with open(
    OUTPUT_FILE,
    "w",
    encoding="utf-8"
) as f:

    f.write(
        "Benchmark Statistics\n"
    )

    f.write(
        "=" * 70 + "\n\n"
    )

    for result in results:

        mean_100 = result["mean"] * 100

        std_100 = result["std"] * 100

        best_100 = result["best"] * 100

        f.write(
            f"File: "
            f"{result['file']}\n"
        )

        f.write(
            f"Run container: "
            f"{result['container']}\n"
        )

        f.write(
            f"Metric: "
            f"{result['metric']}\n"
        )

        f.write(
            f"Number of entries: "
            f"{result['entries']}\n"
        )

        # ----------------------------------------------------
        # Mean
        # ----------------------------------------------------

        f.write(
            f"Mean: "
            f"{result['mean']:.6f}\n"
        )

        f.write(
            f"Variance: "
            f"{result['variance']:.6f}\n"
        )

        f.write(
            f"Standard deviation: "
            f"{result['std']:.6f}\n"
        )

        f.write(
            f"Mean ± Std (%): "
            f"{mean_100:.2f} ± "
            f"{std_100:.2f}\n"
        )

        # ----------------------------------------------------
        # Best
        # ----------------------------------------------------

        f.write(
            f"Best: "
            f"{result['best']:.6f}\n"
        )

        f.write(
            f"Best (%): "
            f"{best_100:.2f}\n"
        )

        f.write(
            "-" * 70 + "\n\n"
        )


# ============================================================
# 完成
# ============================================================

print("=" * 70)

print("Finished!")

print(
    f"Results saved to: "
    f"{OUTPUT_FILE}"
)

print("=" * 70)