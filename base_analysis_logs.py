import os
import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# ==================== 配置 ====================
BASE_DIR = "基础测试"
RESULT_DIR = "统计结果"
TIME_LIMIT = 10800
TOL = 10.0
SYSTEM_IDS = [1, 2, 3, 4, 5, 6]

MODEL_FOLDERS = [
    "1. UC-SSC",
    "2. MEG-NoLink",
    "3. TUC-SSC2-NoLink",
    "4. MEG",
    "5. TUC-SSC1",
    "6. TUC-SSC2",
]
MODEL_NAMES = [
    "UC-SSC",
    "MEG-NoLink",
    "TUC-SSC2-NoLink",
    "MEG",
    "TUC-SSC1",
    "TUC-SSC2",
]
BENCHMARK_MODEL = "UC-SSC"

plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = ['Arial', 'DejaVu Sans']
sns.set_theme(style="whitegrid")

# ==================== 正则表达式 ====================
class LogPatterns:
    DAY_START = re.compile(r"(?:开始计算第|---\s*第)\s*(\d+)\s*(?:天|---)")
    DAY_END = re.compile(r"\[CompRes 写入\]")
    PRESOLVE_TIME = re.compile(r"Presolve time:\s*([\d\.]+)s")
    ROOT_RELAX = re.compile(r"Root relaxation: objective\s*([+\-\d\.eE]+),.*,\s*([\d\.]+)\s*seconds")
    BB_SEARCH = re.compile(r"Explored\s*(\d+)\s*nodes.*in\s*([\d\.]+)\s*seconds")
    CPU_TIME = re.compile(r"求解完成.*求解用时：\s*([\d\.]+)\s*秒")
    MIP_HEADER = re.compile(r"Expl Unexpl")
    MIP_ROW = re.compile(r"(\d+\.\d+)%\s+[-]*\s+(\d+)s")
    OBJ_LINE = re.compile(
        r"Best objective\s+([+-]?\d*\.?\d+(?:[eE][+-]?\d+)?),"
        r"\s+best bound\s+([+-]?\d*\.?\d+(?:[eE][+-]?\d+)?),"
        r"\s+gap\s+([\d\.]+)%"
    )
    FIRST_FEASIBLE = re.compile(
        r".*?"
        r"([\d\.eE\+\-]+)\s+"
        r"([\d\.eE\+\-]+)\s+"
        r"(\d+(?:\.\d+)?)%\s+"
        r".*?"
        r"(\d+)s\s*$",
        re.IGNORECASE
    )
    COPTPROB = re.compile(r"COPTPROB:\s*(\d+)\s+rows,\s*(\d+)\s+columns,\s*(\d+)\s+nonzeros")
    VARIABLE_TYPES = re.compile(r"Variable types:\s*(\d+)\s+continuous,\s*(\d+)\s+integer")

# ==================== 解析函数 ====================
def extract_metrics(block, model_name):
    metrics = {
        "Model": model_name,
        "HasSolution": False,
        "Presolve_T": np.nan,
        "RootRelax_T": np.nan,
        "RootRelax_Val": np.nan,
        "BB_T": np.nan,
        "Nodes": np.nan,
        "CPU_T": np.nan,
        "Gap_Curve": [],
        "ObjVal": np.nan,
        "BestBound": np.nan,
        "FinalGap": np.nan,
        "FirstFeasibleGap": np.nan,
        "FirstFeasibleTime": np.nan,
        "Rows": np.nan,
        "Columns": np.nan,
        "Nonzeros": np.nan,
        "ContVars": np.nan,
        "IntVars": np.nan,
    }

    if m := LogPatterns.PRESOLVE_TIME.search(block):
        metrics["Presolve_T"] = float(m.group(1))
    if m := LogPatterns.ROOT_RELAX.search(block):
        metrics["RootRelax_Val"] = float(m.group(1))
        metrics["RootRelax_T"] = float(m.group(2))
    if m := LogPatterns.BB_SEARCH.search(block):
        metrics["Nodes"] = int(m.group(1))
        metrics["BB_T"] = float(m.group(2))
    if m := LogPatterns.CPU_TIME.search(block):
        metrics["CPU_T"] = float(m.group(1))
        if metrics["CPU_T"] < TIME_LIMIT:   # 成功求解：CPU时间 < 10800秒
            metrics["HasSolution"] = True

    if m := LogPatterns.OBJ_LINE.search(block):
        metrics["ObjVal"] = float(m.group(1))
        metrics["BestBound"] = float(m.group(2))
        metrics["FinalGap"] = float(m.group(3)) / 100.0

    if m := LogPatterns.COPTPROB.search(block):
        metrics["Rows"] = int(m.group(1))
        metrics["Columns"] = int(m.group(2))
        metrics["Nonzeros"] = int(m.group(3))
    if m := LogPatterns.VARIABLE_TYPES.search(block):
        metrics["ContVars"] = int(m.group(1))
        metrics["IntVars"] = int(m.group(2))

    lines = block.split('\n')
    for line in lines:
        if "Incumbent" in line or "Expl" in line or line.strip() == "":
            continue
        m = LogPatterns.FIRST_FEASIBLE.search(line)
        if m:
            try:
                _ = float(m.group(1))
                _ = float(m.group(2))
                gap_percent = float(m.group(3))
                elapsed = int(m.group(4))
                metrics["FirstFeasibleGap"] = gap_percent / 100.0
                metrics["FirstFeasibleTime"] = elapsed
                break
            except ValueError:
                continue

    in_table = False
    gap_points = []
    for line in lines:
        if LogPatterns.MIP_HEADER.search(line):
            in_table = True
            continue
        if in_table and "Best objective" in line:
            break
        if in_table:
            m = LogPatterns.MIP_ROW.search(line)
            if m:
                gap_points.append((int(m.group(2)), float(m.group(1))))
    metrics["Gap_Curve"] = gap_points

    if (pd.isna(metrics["RootRelax_Val"]) and
        pd.isna(metrics["Rows"]) and
        pd.isna(metrics["CPU_T"]) and
        pd.isna(metrics["ObjVal"])):
        return None
    return metrics

def parse_log_file(filepath, model_name):
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            content = f.read()
    except:
        with open(filepath, 'r', encoding='gbk') as f:
            content = f.read()

    day_parts = LogPatterns.DAY_START.split(content)
    records = []
    for i in range(1, len(day_parts), 2):
        day_id = int(day_parts[i])
        day_text = day_parts[i+1]
        if m := LogPatterns.DAY_END.search(day_text):
            day_text = day_text[:m.start()]
        metrics = extract_metrics(day_text, model_name)
        if metrics:
            metrics["Day"] = day_id
            records.append(metrics)

    df = pd.DataFrame(records)
    if df.empty:
        return df
    df = df.sort_values("Day").reset_index(drop=True)
    return df

def recompute_bb(df):
    df = df.copy()
    df["Presolve_T"] = df["Presolve_T"].fillna(0.0)
    df["RootRelax_T"] = df["RootRelax_T"].fillna(0.0)
    df["BB_T"] = df["CPU_T"] - df["Presolve_T"] - df["RootRelax_T"]
    df["BB_T"] = df["BB_T"].clip(lower=0.0)
    return df

# ==================== 汇总表格生成 ====================
def summary_by_system(df_all, metric_list, agg_func='mean'):
    agg = df_all.groupby(["System", "Model"])[metric_list].agg(agg_func).reset_index()
    pivot = agg.pivot(index="System", columns="Model")
    pivot = pivot.reorder_levels([1,0], axis=1).sort_index(axis=1)
    return pivot

def summary_overall(df_all, metric_list, agg_func='mean'):
    """总体平均（所有系统合并，即120个case）"""
    agg = df_all.groupby("Model")[metric_list].agg(agg_func).reset_index()
    return agg.set_index("Model").T

def summary_fastest_nodes(df_all):
    df_ok = df_all[df_all["HasSolution"]]
    records = []
    for s in SYSTEM_IDS:
        dsys = df_ok[df_ok["System"] == s]
        cnt = {m: 0 for m in MODEL_NAMES}
        unique_cnt = {m: 0 for m in MODEL_NAMES}
        for day in dsys["Day"].unique():
            d = dsys[dsys["Day"] == day]
            valid = d[d["CPU_T"].notna()]
            if valid.empty:
                continue
            mn = valid["CPU_T"].min()
            winners = valid[valid["CPU_T"] <= mn + TOL]["Model"].tolist()
            strict_winners = [w for w in winners if valid[valid["CPU_T"] < mn + 1e-6]["Model"].tolist() == [w]]
            for w in winners:
                cnt[w] += 1
            for w in strict_winners:
                unique_cnt[w] += 1
        node_avg = dsys.groupby("Model")["Nodes"].mean().to_dict()
        for m in MODEL_NAMES:
            records.append({
                "System": s,
                "Model": m,
                "Fastest_Count": cnt[m],
                "Unique_Wins": unique_cnt[m],
                "Avg_Nodes": node_avg.get(m, np.nan)
            })
    df = pd.DataFrame(records)
    return df.pivot(index="System", columns="Model", values=["Fastest_Count", "Unique_Wins", "Avg_Nodes"])

# ==================== 绘图函数（最终修正版） ====================
def plot_computation_time(pivot, save_dir):
    phase_map = {"CPU_T":"Total CPU", "Presolve_T":"Presolve",
                 "RootRelax_T":"Root Relax", "BB_T":"B&B"}
    for sys in SYSTEM_IDS:
        if sys not in pivot.index:
            continue
        df_sys = pivot.loc[sys]
        df_wide = df_sys.unstack(level=0).reset_index().rename(columns={'index': 'Phase'})
        df_long = df_wide.melt(id_vars="Phase", var_name="Model", value_name="Time")
        df_long = df_long.dropna(subset=["Time"])
        df_long["Phase"] = df_long["Phase"].map(phase_map)
        if df_long.empty:
            continue
        plt.figure(figsize=(10,6))
        sns.barplot(data=df_long, x="Model", y="Time", hue="Phase")
        plt.title(f"System {sys} - Average Computation Time")
        plt.ylabel("Time (s)")
        plt.xticks(rotation=45, ha="right")
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, f"CompTime_System{sys}.png"), dpi=300)
        plt.close()

def plot_fastest_count(pivot, save_dir):
    try:
        data = pivot.xs('Fastest_Count', axis=1, level=0)
    except KeyError:
        print("Fastest_Count data not found.")
        return
    if data.empty:
        return
    data = data.astype(float)
    data.plot(kind="bar", stacked=True, figsize=(10,6))
    plt.title("Fastest Count per System (10s tolerance)")
    plt.ylabel("Number of Days (out of 20)")
    plt.xlabel("System")
    plt.xticks(rotation=0)
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "Fastest_Count.png"), dpi=300)
    plt.close()

def plot_first_gap(pivot, save_dir):
    try:
        gaps = pivot.xs('FirstFeasibleGap', axis=1, level=0)
        times = pivot.xs('FirstFeasibleTime', axis=1, level=0)
    except KeyError:
        print("FirstFeasible data not found.")
        return
    for sys in SYSTEM_IDS:
        if sys not in gaps.index:
            continue
        gap_row = gaps.loc[sys].dropna()
        time_row = times.loc[sys].dropna()
        common = gap_row.index.intersection(time_row.index)
        if common.empty:
            continue
        data = pd.DataFrame({
            "Model": common,
            "FirstFeasibleGap": gap_row.loc[common].values,
            "FirstFeasibleTime": time_row.loc[common].values
        })
        plt.figure(figsize=(8,6))
        sns.scatterplot(data=data, x="FirstFeasibleTime", y="FirstFeasibleGap", hue="Model", s=100)
        plt.title(f"System {sys} - First Feasible Gap vs Time")
        plt.xlabel("Time (s)")
        plt.ylabel("Gap (%)")
        plt.legend(bbox_to_anchor=(1.05,1), loc="upper left")
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, f"FirstGap_System{sys}.png"), dpi=300)
        plt.close()

def plot_model_size(pivot, save_dir):
    metrics = ["Rows", "Columns", "Nonzeros", "ContVars", "IntVars"]
    for sys in SYSTEM_IDS:
        if sys not in pivot.index:
            continue
        df_sys = pivot.loc[sys]
        df_wide = df_sys.unstack(level=0).reset_index().rename(columns={'index': 'Metric'})
        df_long = df_wide.melt(id_vars="Metric", var_name="Model", value_name="Value")
        df_long = df_long[df_long["Metric"].isin(metrics)].dropna(subset=["Value"])
        if df_long.empty:
            continue
        plt.figure(figsize=(10,6))
        sns.barplot(data=df_long, x="Model", y="Value", hue="Metric")
        plt.yscale("log")
        plt.title(f"System {sys} - Model Size (log scale)")
        plt.ylabel("Count (log)")
        plt.xticks(rotation=45, ha="right")
        plt.legend(bbox_to_anchor=(1.05,1), loc="upper left")
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, f"ModelSize_System{sys}.png"), dpi=300)
        plt.close()

# ==================== 主程序 ====================
def main():
    os.makedirs(RESULT_DIR, exist_ok=True)
    all_dfs = []

    for sys_id in SYSTEM_IDS:
        print(f"\n>>> 处理 System {sys_id}")
        system_records = []
        for folder, model in zip(MODEL_FOLDERS, MODEL_NAMES):
            folder_path = os.path.join(BASE_DIR, folder)
            if not os.path.exists(folder_path):
                continue
            target_prefix = f"System{sys_id}-"
            log_files = [f for f in os.listdir(folder_path) if f.startswith(target_prefix) and f.endswith('.log')]
            if not log_files:
                continue
            filepath = os.path.join(folder_path, log_files[0])
            df_day = parse_log_file(filepath, model)
            if df_day.empty:
                continue
            df_day["System"] = sys_id
            system_records.append(df_day)
        if system_records:
            df_sys = pd.concat(system_records, ignore_index=True)
            df_sys = recompute_bb(df_sys)
            all_dfs.append(df_sys)

    if not all_dfs:
        print("未提取到任何数据，请检查路径。")
        return

    df_all = pd.concat(all_dfs, ignore_index=True)

    print("\n生成汇总表格...")
    out_csv_dir = os.path.join(RESULT_DIR, "Tables")
    os.makedirs(out_csv_dir, exist_ok=True)

    size_metrics = ["Rows", "Columns", "Nonzeros", "ContVars", "IntVars"]
    time_metrics = ["CPU_T", "Presolve_T", "RootRelax_T", "BB_T"]
    relax_metrics = ["RootRelax_Val"]
    node_metrics = ["Nodes"]
    first_gap_metrics = ["FirstFeasibleGap", "FirstFeasibleTime"]

    # ---- 系统平均（6个系统分别平均） ----
    pivot_size = summary_by_system(df_all, size_metrics)
    pivot_size.to_csv(os.path.join(out_csv_dir, "System_Avg_Model_Size.csv"))

    pivot_time = summary_by_system(df_all, time_metrics)
    pivot_time.to_csv(os.path.join(out_csv_dir, "System_Avg_Computation_Time.csv"))

    pivot_relax = summary_by_system(df_all, relax_metrics)
    pivot_relax.to_csv(os.path.join(out_csv_dir, "System_Avg_RootRelax.csv"))

    pivot_nodes = summary_by_system(df_all, node_metrics)
    pivot_nodes.to_csv(os.path.join(out_csv_dir, "System_Avg_Nodes.csv"))

    pivot_first = summary_by_system(df_all, first_gap_metrics)
    pivot_first.to_csv(os.path.join(out_csv_dir, "System_Avg_FirstGap.csv"))

    # ---- 整体平均（120个case整体平均） ----
    overall_size = summary_overall(df_all, size_metrics)
    overall_size.to_csv(os.path.join(out_csv_dir, "Overall_Avg_Model_Size.csv"))

    overall_time = summary_overall(df_all, time_metrics)
    overall_time.to_csv(os.path.join(out_csv_dir, "Overall_Avg_Computation_Time.csv"))

    overall_relax = summary_overall(df_all, relax_metrics)
    overall_relax.to_csv(os.path.join(out_csv_dir, "Overall_Avg_RootRelax.csv"))

    overall_nodes = summary_overall(df_all, node_metrics)
    overall_nodes.to_csv(os.path.join(out_csv_dir, "Overall_Avg_Nodes.csv"))

    overall_first = summary_overall(df_all, first_gap_metrics)
    overall_first.to_csv(os.path.join(out_csv_dir, "Overall_Avg_FirstGap.csv"))

    # ---- 求解成功率（按系统、模型） ----
    success = df_all.groupby(["System", "Model"]).agg(
        Solved_Count=("HasSolution", lambda x: x.sum()),
        Total_Days=("HasSolution", "count")
    ).reset_index()
    success["Success_Rate"] = success["Solved_Count"] / success["Total_Days"]
    success.to_csv(os.path.join(out_csv_dir, "Solve_Success_Rate.csv"), index=False)

    # ---- 整体成功率（所有系统合并） ----
    overall_success = df_all.groupby("Model").agg(
        Solved_Count=("HasSolution", lambda x: x.sum()),
        Total_Days=("HasSolution", "count")
    ).reset_index()
    overall_success["Success_Rate"] = overall_success["Solved_Count"] / overall_success["Total_Days"]
    overall_success.to_csv(os.path.join(out_csv_dir, "Overall_Solve_Success_Rate.csv"), index=False)

    # ---- 最快次数统计 ----
    pivot_fastest = summary_fastest_nodes(df_all)
    pivot_fastest.to_csv(os.path.join(out_csv_dir, "Fastest_Count_Nodes.csv"))

    # ---- 打印整体平均结果（便于检查） ----
    print("\n===== 整体平均（120个case）=====")
    print("模型规模 (Rows, Columns, Nonzeros, ContVars, IntVars):")
    print(overall_size)
    print("\n计算时间 (CPU, Presolve, RootRelax, BB):")
    print(overall_time)
    print("\n根松弛值:")
    print(overall_relax)
    print("\nB&B节点数:")
    print(overall_nodes)
    print("\n首次可行解 (Gap, Time):")
    print(overall_first)
    print("\n整体求解成功率:")
    print(overall_success)

    print("\n生成图表...")
    plot_dir = os.path.join(RESULT_DIR, "Figures")
    os.makedirs(plot_dir, exist_ok=True)

    plot_computation_time(pivot_time, plot_dir)
    plot_fastest_count(pivot_fastest, plot_dir)
    plot_first_gap(pivot_first, plot_dir)
    plot_model_size(pivot_size, plot_dir)

    plt.figure(figsize=(10,6))
    pivot_relax.unstack().plot(kind="bar")
    plt.title("Average Root Relaxation Value per System")
    plt.ylabel("Objective Value")
    plt.xticks(rotation=0)
    plt.tight_layout()
    plt.savefig(os.path.join(plot_dir, "RootRelax_Comparison.png"), dpi=300)
    plt.close()

    print(f"\n所有结果已保存至 {RESULT_DIR}，其中表格在 Tables/，图表在 Figures/")

if __name__ == "__main__":
    main()