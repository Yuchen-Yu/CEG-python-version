import argparse
import pandas as pd
import numpy as np
import ruptures as rpt
import matplotlib.pyplot as plt


def main():
    parser = argparse.ArgumentParser(
        description="Process perturbation and extinction data from .txt file."
    )
    parser.add_argument("input", help="Input .txt file")
    parser.add_argument("--output_prefix", default="ms1", help="Output file prefix (default: ms1)")
    parser.add_argument("--plot", action="store_true", help="Enable plotting (default: off)")
    parser.add_argument("--segments", type=int, default=100, help="Number of segments for changepoint analysis")
    parser.add_argument("--bkps", type=int, default=10, help="Number of changepoints per segment")
    args = parser.parse_args()

    # 1. 读取 txt 文件，提取倒数第3列和倒数第1列
    with open(args.input, "r") as f:
        lines = [line.strip().split() for line in f if line.strip()]

    df = pd.DataFrame(lines)
    df = df.iloc[:, [-3, -1]]  # 倒数第3列和倒数第1列
    df.columns = ["pertubation", "secondary_extinction"]

    # 转为数值型
    df = df.apply(pd.to_numeric, errors="coerce")

    # 2. 去除前4行
    df = df.iloc[4:].reset_index(drop=True)

    # 3. 保存为 CSV
    csv_path = f"{args.output_prefix}.csv"
    df.to_csv(csv_path, index=False)
    print(f"[✔] Saved cleaned data to {csv_path}")

    # 4. changepoint 分析（替代 changepoint.np）
    n_rows = len(df)
    s = n_rows // args.segments
    n_segments = args.bkps

    ms1cp_np = np.full((args.segments, n_segments), np.nan)

    for i in range(args.segments):
        subset = df["secondary_extinction"].iloc[s * i : s * (i + 1)].to_numpy()
        if len(subset) > n_segments:
            algo = rpt.KernelCPD(kernel="linear").fit(subset)
            result = algo.predict(n_bkps=n_segments)
            result = result[:-1]  # 去掉最后一个结尾点
            pert_values = [df["pertubation"].iloc[min(idx + s * i, n_rows - 1)] for idx in result]
            ms1cp_np[i, :len(pert_values)] = pert_values

    cp_csv_path = f"{args.output_prefix}cp_np.csv"
    pd.DataFrame(ms1cp_np).to_csv(cp_csv_path, index=False, header=False)
    print(f"[✔] Saved changepoint results to {cp_csv_path}")

    # 5. 提取不同阈值下的最小 pertubation
    thresholds = [0.1, 0.2, 0.3, 0.4]
    selected_rows = pd.DataFrame()

    for t in thresholds:
        subset = df[df["pertubation"] > t]
        if not subset.empty:
            min_val = subset["pertubation"].min()
            selected_rows = pd.concat([selected_rows, df[df["pertubation"] == min_val]])

    selp_csv_path = f"{args.output_prefix}selp.csv"
    selected_rows.to_csv(selp_csv_path, index=False)
    print(f"[✔] Saved selected perturbation data to {selp_csv_path}")

    # 6. 绘图部分（对应 R 的 plot）
    if args.plot:
        plt.figure(figsize=(8, 6))
        plt.scatter(
            df["pertubation"], df["secondary_extinction"],
            c=(1, 0, 0, 0.8), s=40, edgecolors="none"
        )

        plt.xlabel("Pertubation")
        plt.ylabel("Secondary extinction")
        plt.grid(True, linestyle="--", alpha=0.4)
        plt.tight_layout()
        plot_path = f"{args.output_prefix}_plot.png"
        plt.savefig(plot_path, dpi=300)
        print(f"[✔] Plot saved as {plot_path}")


if __name__ == "__main__":
    main()
