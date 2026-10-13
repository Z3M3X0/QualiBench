import json
import glob
import os


# ============================================================
# 配置
# ============================================================

# 当前 Python 文件所在文件夹
INPUT_DIR = os.path.dirname(os.path.abspath(__file__))

# 最终结果文件：与 Python 文件放在同一个文件夹
OUTPUT_FILE = os.path.join(INPUT_DIR, "results.txt")


# ============================================================
# 计算单个 JSONL 文件
# ============================================================

def calculate_file(input_file):

    data = []

    # ---------- 读取 JSONL ----------
    with open(input_file, "r", encoding="utf-8") as f:

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

    # ---------- 每个 idx 的 Mean 和 Best ----------
    idx_means = []
    idx_bests = []

    for item in data:

        idx = item.get("idx")

        # ---------- 检查 runs ----------
        if "runs" not in item:

            print(
                f"[WARNING] {os.path.basename(input_file)} "
                f"idx={idx} 缺少 'runs' 字段，跳过"
            )

            continue

        runs = item["runs"]

        # ---------- 获取三个 similarity ----------
        similarities = []

        for run_name in [
            "run_1_initial",
            "run_2_repeat_1",
            "run_3_repeat_2"
        ]:

            if (
                run_name in runs
                and "similarity" in runs[run_name]
            ):

                similarities.append(
                    runs[run_name]["similarity"]
                )

        # 必须正好有 3 个 similarity
        if len(similarities) != 3:

            print(
                f"[WARNING] {os.path.basename(input_file)} "
                f"idx={idx} 有 {len(similarities)} 个 similarity，跳过"
            )

            continue

        # ====================================================
        # 每个 idx
        # ====================================================

        # 1. 三次结果的平均值
        idx_mean = sum(similarities) / 3

        # 2. 三次结果中最好的 similarity
        idx_best = max(similarities)

        idx_means.append(idx_mean)
        idx_bests.append(idx_best)

    # ---------- 检查数据 ----------
    N = len(idx_means)

    if N == 0:
        return None

    # ========================================================
    # Mean similarity
    # ========================================================

    overall_mean = sum(idx_means) / N

    # ---------- 总体方差 ----------
    variance = sum(
        (x - overall_mean) ** 2
        for x in idx_means
    ) / N

    # ---------- 标准差 ----------
    std = variance ** 0.5

    # ========================================================
    # Best similarity
    # ========================================================

    overall_best = sum(idx_bests) / N

    return {
        "file": os.path.basename(input_file),
        "entries": N,

        # Mean
        "mean": overall_mean,
        "variance": variance,
        "std": std,

        # Best
        "best": overall_best
    }


# ============================================================
# 批量处理
# ============================================================

jsonl_files = sorted(
    glob.glob(
        os.path.join(INPUT_DIR, "*.jsonl")
    )
)

if not jsonl_files:

    raise FileNotFoundError(
        f"在 {INPUT_DIR} 中没有找到 JSONL 文件"
    )


print("=" * 70)
print(f"Found {len(jsonl_files)} JSONL files")
print("=" * 70)


results = []


for input_file in jsonl_files:

    print(
        f"Processing: "
        f"{os.path.basename(input_file)}"
    )

    result = calculate_file(input_file)

    if result is not None:

        results.append(result)

        # ====================================================
        # Mean × 100 ± Std × 100
        # ====================================================

        mean_100 = result["mean"] * 100
        std_100 = result["std"] * 100

        # ====================================================
        # Best × 100
        # ====================================================

        best_100 = result["best"] * 100

        print(
            f"  N                = {result['entries']}"
        )

        print(
            f"  Mean             = {result['mean']:.6f}"
        )

        print(
            f"  Variance         = {result['variance']:.6f}"
        )

        print(
            f"  Std              = {result['std']:.6f}"
        )

        print(
            f"  Mean ± Std (%)   = "
            f"{mean_100:.2f} ± {std_100:.2f}"
        )

        print(
            f"  Best similarity  = "
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
        "Similarity Statistics\n"
    )

    f.write(
        "=" * 70 + "\n\n"
    )

    for result in results:

        # ---------- Mean × 100 ----------
        mean_100 = result["mean"] * 100
        std_100 = result["std"] * 100

        # ---------- Best × 100 ----------
        best_100 = result["best"] * 100

        f.write(
            f"File: {result['file']}\n"
        )

        f.write(
            f"Number of entries: "
            f"{result['entries']}\n"
        )

        # ====================================================
        # Mean
        # ====================================================

        f.write(
            f"Mean similarity: "
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
            f"{mean_100:.2f} ± {std_100:.2f}\n"
        )

        # ====================================================
        # Best
        # ====================================================

        f.write(
            f"Best similarity: "
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
print(f"Results saved to: {OUTPUT_FILE}")
print("=" * 70)