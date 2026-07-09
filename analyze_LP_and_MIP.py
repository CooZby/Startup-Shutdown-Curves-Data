import os
import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# ==================== 配置 ====================
LP_BASE = "LP测试"
MIP_BASE = "基础测试"
RESULT_DIR = "综合统计结果"

# 六个模型（LP和MIP使用相同文件夹命名）
MODEL_FOLDERS = [
    "1. UC-SSC",
    "2. MEG-NoLink",
    "3. TUC-SSC2-NoLink",
    "4. MEG",
    "5. TUC-SSC1",
    "6. TUC-SSC2"
]
MODEL_NAMES = ["UC-SSC", "MEG-NoLink", "TUC-SSC2-NoLink", "MEG", "TUC-SSC1", "TUC-SSC2"]

BENCHMARK = "UC-SSC"
SYSTEM_IDS = [1, 2, 3, 4, 5, 6]
COLORS = {
    "UC-SSC": "#1f77b4",
    "MEG-NoLink": "#ff7f0e",
    "TUC-SSC2-NoLink": "#2ca02c",
    "MEG": "#9467bd",
    "TUC-SSC1": "#d62728",
    "TUC-SSC2": "#8c564b"
}
plt.rcParams['font.family'] = 'Times New Roman'
plt.rcParams['font.size'] = 12
sns.set_theme(style="whitegrid")


# ==================== 正则（LP专用） ====================
class LP_Patterns:
    DAY_START = re.compile(r"---\s*第\s*(\d+)\s*天\s*---")
    CPU_TIME = re.compile(r"求解完成.*求解用时：([\d\.]+)\s+秒")
    OBJ_LINE = re.compile(r"Optimal objective\s+([+\-\d\.eE]+)")


# ==================== 解析LP日志 ====================
def parse_lp_log(filepath, model_name, verbose=False):
    if verbose:
        print(f"  解析LP: {filepath}")
    with open(filepath, 'r', encoding='utf-8') as f:
        content = f.read()
    day_parts = LP_Patterns.DAY_START.split(content)
    if len(day_parts) <= 1:
        if verbose:
            print(f"    警告: 未找到天标记")
        return pd.DataFrame()
    records = []
    for i in range(1, len(day_parts), 2):
        day = int(day_parts[i])
        block = day_parts[i + 1]
        cpu_m = LP_Patterns.CPU_TIME.search(block)
        cpu = float(cpu_m.group(1)) if cpu_m else np.nan
        obj_m = LP_Patterns.OBJ_LINE.search(block)
        obj = float(obj_m.group(1)) if obj_m else np.nan
        if pd.isna(obj):
            if verbose:
                print(f"    警告: 第{day}天未找到Optimal objective")
        records.append({"Day": day, "CPU_T": cpu, "ObjVal": obj})
    df = pd.DataFrame(records)
    if df.empty:
        return df
    df["Model"] = model_name
    return df.sort_values("Day").reset_index(drop=True)


# ==================== 解析MIP日志（根节点松弛值+时间） ====================
def parse_mip_log(filepath, model_name, verbose=False):
    if verbose:
        print(f"  解析MIP: {filepath}")
    with open(filepath, 'r', encoding='utf-8') as f:
        content = f.read()
    day_parts = re.split(r'(?:开始计算第|---\s*第)\s*(\d+)\s*(?:天|---)', content)
    if len(day_parts) <= 1:
        if verbose:
            print(f"    警告: 未找到天标记")
        return pd.DataFrame()
    records = []
    for i in range(1, len(day_parts), 2):
        day = int(day_parts[i])
        block = day_parts[i + 1]
        m = re.search(r"\[CompRes 写入\]", block)
        if m:
            block = block[:m.start()]
        m = re.search(r"Root relaxation: objective\s*([+\-\d\.eE]+),.*,\s*([\d\.]+)\s*seconds", block)
        if m:
            root_val = float(m.group(1))
            root_time = float(m.group(2))
        else:
            root_val = np.nan
            root_time = np.nan
            if verbose:
                print(f"    警告: 第{day}天未找到Root relaxation")
        m = re.search(r"求解完成.*求解用时：\s*([\d\.]+)\s*秒", block)
        cpu = float(m.group(1)) if m else np.nan
        records.append({"Day": day, "CPU_T": cpu, "RootRelax_Val": root_val, "RootRelax_T": root_time})
    df = pd.DataFrame(records)
    if df.empty:
        return df
    df["Model"] = model_name
    return df.sort_values("Day").reset_index(drop=True)


# ==================== 扫描文件夹（通用） ====================
def scan_folders(base_dir, folder_list, model_names, parser_func, verbose=False):
    all_data = []
    for folder, model in zip(folder_list, model_names):
        folder_path = os.path.join(base_dir, folder)
        if not os.path.exists(folder_path):
            print(f"错误: 文件夹 {folder_path} 不存在！")
            continue
        for sys in SYSTEM_IDS:
            prefix = f"System{sys}-"
            files = [f for f in os.listdir(folder_path) if f.startswith(prefix) and f.endswith('.log')]
            if not files:
                print(f"警告: {folder} 中未找到 {prefix}*.log")
                continue
            filepath = os.path.join(folder_path, files[0])
            df = parser_func(filepath, model, verbose)
            if df.empty:
                print(f"警告: {filepath} 解析失败")
                continue
            df["System"] = sys
            all_data.append(df)
    if all_data:
        return pd.concat(all_data, ignore_index=True)
    return pd.DataFrame()


# ==================== 统计函数 ====================
def compute_summary(df, metric_cols):
    """计算每个系统每个模型的平均、最小、最大"""
    summary = df.groupby(["System", "Model"])[metric_cols].agg(["mean", "min", "max"]).reset_index()
    cols = []
    for metric in metric_cols:
        for stat in ["mean", "min", "max"]:
            cols.append((metric, stat))
    summary.columns = ["System", "Model"] + [f"{m}_{s}" for m, s in cols]
    return summary


def add_improvement_ratio(summary, bench_model, obj_col="ObjVal"):
    """计算提升倍数：(current - bench) / bench * 100，正值表示扩大（更差），但您需要的是正值表示提升（紧致）？我们按公式计算"""
    bench = summary[summary["Model"] == bench_model][["System", f"{obj_col}_mean"]].rename(
        columns={f"{obj_col}_mean": "Bench_Obj"})
    merged = pd.merge(summary, bench, on="System", how="left")
    # 提升倍数百分比 (current - bench) / bench * 100
    merged[f"Improve_{obj_col}_mean"] = (merged[f"{obj_col}_mean"] - merged["Bench_Obj"]) / merged["Bench_Obj"] * 100
    merged[f"Improve_{obj_col}_min"] = (merged[f"{obj_col}_min"] - merged["Bench_Obj"]) / merged["Bench_Obj"] * 100
    merged[f"Improve_{obj_col}_max"] = (merged[f"{obj_col}_max"] - merged["Bench_Obj"]) / merged["Bench_Obj"] * 100
    merged.loc[merged["Model"] == bench_model, [f"Improve_{obj_col}_mean", f"Improve_{obj_col}_min",
                                                f"Improve_{obj_col}_max"]] = 0.0
    return merged


def overall_stats(summary, metric_cols):
    """每个模型在所有系统上的整体平均、最小、最大"""
    result = pd.DataFrame()
    for col in metric_cols:
        agg = summary.groupby("Model")[col].agg(["mean", "min", "max"]).reset_index()
        agg.columns = ["Model", f"{col}_overall_mean", f"{col}_overall_min", f"{col}_overall_max"]
        if result.empty:
            result = agg
        else:
            result = pd.merge(result, agg, on="Model")
    return result


# ==================== 绘图函数 ====================
def plot_comparison(df_summary, metric, ylabel, title_prefix, save_dir, model_order=None):
    if model_order is None:
        model_order = sorted(df_summary["Model"].unique())
    for sys in SYSTEM_IDS:
        df_sys = df_summary[df_summary["System"] == sys]
        if df_sys.empty:
            continue
        existing = df_sys["Model"].unique()
        order = [m for m in model_order if m in existing]
        df_sys = df_sys.set_index("Model").reindex(order).reset_index()
        err_lower = df_sys[f"{metric}_mean"] - df_sys[f"{metric}_min"]
        err_upper = df_sys[f"{metric}_max"] - df_sys[f"{metric}_mean"]
        err_lower = err_lower.fillna(0).clip(lower=0)
        err_upper = err_upper.fillna(0).clip(lower=0)
        yerr = [err_lower, err_upper]
        ax = df_sys.plot(kind="bar", x="Model", y=f"{metric}_mean", yerr=yerr,
                         capsize=4, figsize=(8, 5), legend=False,
                         color=[COLORS.get(m, "gray") for m in df_sys["Model"]])
        ax.set_title(f"{title_prefix} - System {sys}")
        ax.set_ylabel(ylabel)
        ax.set_xlabel("Model")
        ax.set_xticklabels(df_sys["Model"], rotation=45, ha="right")
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, f"{title_prefix.replace(' ', '_')}_System{sys}.png"), dpi=300)
        plt.close()


def plot_overall(df_overall, time_col, obj_col, ylabel_left, ylabel_right, save_dir, filename):
    df_plot = df_overall.set_index("Model")
    fig, ax1 = plt.subplots(figsize=(8, 5))
    ax1.bar(df_plot.index, df_plot[time_col], color=[COLORS.get(m, "gray") for m in df_plot.index], alpha=0.7)
    ax1.set_ylabel(ylabel_left, color="blue")
    ax1.tick_params(axis='y', labelcolor="blue")
    ax2 = ax1.twinx()
    ax2.plot(df_plot.index, df_plot[obj_col], marker='o', color='red', linestyle='-', linewidth=2)
    ax2.set_ylabel(ylabel_right, color="red")
    ax2.tick_params(axis='y', labelcolor="red")
    plt.title("Overall Average (all systems)")
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, filename), dpi=300)
    plt.close()


# ==================== 主程序 ====================
def main():
    os.makedirs(RESULT_DIR, exist_ok=True)

    # ---------- 处理 LP ----------
    print("扫描LP测试文件夹...")
    df_lp = scan_folders(LP_BASE, MODEL_FOLDERS, MODEL_NAMES, parse_lp_log, verbose=True)
    if df_lp.empty:
        print("LP测试未提取到任何数据，请检查文件夹和日志格式。")
    else:
        lp_summary = compute_summary(df_lp, ["CPU_T", "ObjVal"])
        lp_summary = add_improvement_ratio(lp_summary, BENCHMARK, "ObjVal")
        lp_summary.to_csv(os.path.join(RESULT_DIR, "LP_Summary.csv"), index=False)

        lp_overall_stats = overall_stats(lp_summary, ["CPU_T_mean", "ObjVal_mean", "Improve_ObjVal_mean"])
        lp_overall_stats.to_csv(os.path.join(RESULT_DIR, "LP_Overall_Stats.csv"), index=False)

        lp_overall_avg = lp_summary.groupby("Model")[["CPU_T_mean", "ObjVal_mean"]].mean().reset_index()
        lp_overall_avg.to_csv(os.path.join(RESULT_DIR, "LP_Overall_Avg.csv"), index=False)

        plot_dir = os.path.join(RESULT_DIR, "LP_Figures")
        os.makedirs(plot_dir, exist_ok=True)
        plot_comparison(lp_summary, "CPU_T", "Time (s)", "LP CPU Time", plot_dir, MODEL_NAMES)
        plot_comparison(lp_summary, "Improve_ObjVal", "Improvement (%)", "LP Objective Improvement", plot_dir,
                        MODEL_NAMES)
        plot_overall(lp_overall_avg, "CPU_T_mean", "ObjVal_mean", "Avg CPU Time (s)", "Avg Objective Value", plot_dir,
                     "LP_Overall.png")

    # ---------- 处理 MIP ----------
    print("\n扫描MIP测试文件夹...")
    df_mip = scan_folders(MIP_BASE, MODEL_FOLDERS, MODEL_NAMES, parse_mip_log, verbose=True)
    if df_mip.empty:
        print("MIP测试未提取到任何数据，请检查文件夹和日志格式。")
    else:
        mip_summary = compute_summary(df_mip, ["CPU_T", "RootRelax_Val", "RootRelax_T"])
        mip_summary = add_improvement_ratio(mip_summary, BENCHMARK, "RootRelax_Val")
        mip_summary.to_csv(os.path.join(RESULT_DIR, "MIP_Summary.csv"), index=False)

        mip_overall_stats = overall_stats(mip_summary, ["CPU_T_mean", "RootRelax_Val_mean", "RootRelax_T_mean",
                                                        "Improve_RootRelax_Val_mean"])
        mip_overall_stats.to_csv(os.path.join(RESULT_DIR, "MIP_Overall_Stats.csv"), index=False)

        mip_overall_avg = mip_summary.groupby("Model")[["CPU_T_mean", "RootRelax_Val_mean"]].mean().reset_index()
        mip_overall_avg.to_csv(os.path.join(RESULT_DIR, "MIP_Overall_Avg.csv"), index=False)

        plot_dir = os.path.join(RESULT_DIR, "MIP_Figures")
        os.makedirs(plot_dir, exist_ok=True)
        plot_comparison(mip_summary, "CPU_T", "Time (s)", "MIP CPU Time", plot_dir, MODEL_NAMES)
        plot_comparison(mip_summary, "Improve_RootRelax_Val", "Improvement (%)", "MIP Root Relax Improvement", plot_dir,
                        MODEL_NAMES)
        plot_overall(mip_overall_avg, "CPU_T_mean", "RootRelax_Val_mean", "Avg CPU Time (s)", "Avg Root Relax Value",
                     plot_dir, "MIP_Overall.png")

    print(f"\n所有结果已保存至 {RESULT_DIR}")


if __name__ == "__main__":
    main()