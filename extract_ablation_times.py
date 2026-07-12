import re
import pandas as pd
from collections import defaultdict

# ==========================================
# 1. 配置
# ==========================================
LOG_FILE = "消融实验/0624ablation.log"   # 日志文件路径
OUTPUT_CSV = "ablation_time_summary.csv"

# 模型名称列表（按日志中出现顺序）
MODELS = ['seg_range', 'ramp', 'seg_p', 'p_range']
SYSTEM_IDS = ['S1', 'S2', 'S3', 'S4', 'S5', 'S6']  # 实际可能包括

# ==========================================
# 2. 解析函数
# ==========================================
def parse_log(log_path):
    with open(log_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()

    # 数据结构： system -> day -> model -> {Presolve, RootRelax, BB, Total}
    data = defaultdict(lambda: defaultdict(lambda: defaultdict(dict)))

    current_system = None
    current_day = None
    current_model = None

    # 正则模式
    system_pattern = re.compile(r'正在处理系统\s+(S\d+)')
    day_pattern = re.compile(r'正在计算第\s*(\d+)/20\s+天')
    model_pattern = re.compile(r'运行\s+(\S+)[.][.][.]')

    # 时间提取模式
    presolve_pattern = re.compile(r'Presolve time:\s*([\d.]+)s')
    root_relax_pattern = re.compile(r'Root relaxation:.*?,\s*([\d.]+)\s*seconds')
    bb_search_pattern = re.compile(r'Explored\s+\d+\s+nodes.*in\s*([\d.]+)\s*seconds')
    total_time_pattern = re.compile(r'求解完成（状态=\d+），求解用时：\s*([\d.]+)\s+秒')

    for line in lines:
        # 检测系统
        sys_match = system_pattern.search(line)
        if sys_match:
            current_system = sys_match.group(1)
            continue

        # 检测天
        day_match = day_pattern.search(line)
        if day_match:
            current_day = int(day_match.group(1))
            continue

        # 检测模型
        model_match = model_pattern.search(line)
        if model_match:
            current_model = model_match.group(1)
            # 如果模型不在列表中，保留原样，但通常都在
            continue

        # 提取时间（必须在确定了系统、天、模型之后）
        if current_system is not None and current_day is not None and current_model is not None:
            # Presolve
            presolve = presolve_pattern.search(line)
            if presolve:
                data[current_system][current_day][current_model]['Presolve'] = float(presolve.group(1))

            # Root relaxation
            root = root_relax_pattern.search(line)
            if root:
                data[current_system][current_day][current_model]['RootRelax'] = float(root.group(1))

            # BB search
            bb = bb_search_pattern.search(line)
            if bb:
                data[current_system][current_day][current_model]['BB'] = float(bb.group(1))

            # Total time
            total = total_time_pattern.search(line)
            if total:
                data[current_system][current_day][current_model]['Total'] = float(total.group(1))

    return data

# ==========================================
# 3. 计算平均值
# ==========================================
def compute_averages(data):
    """
    返回一个 DataFrame，索引为 System，列为 (Model, TimeType)
    """
    records = []
    for system, days in data.items():
        for model in MODELS:   # 确保所有模型都有行
            # 收集所有天中该模型的时间
            times = {'Total': [], 'Presolve': [], 'RootRelax': [], 'BB': []}
            for day, models in days.items():
                if model in models:
                    for ttype in times.keys():
                        if ttype in models[model]:
                            times[ttype].append(models[model][ttype])
            # 计算平均值（如果某天缺失该模型，则忽略，但一般每天都有）
            row = {'System': system, 'Model': model}
            for ttype, vals in times.items():
                if vals:
                    row[ttype] = sum(vals) / len(vals)
                else:
                    row[ttype] = None
            records.append(row)
    df = pd.DataFrame(records)
    return df

# ==========================================
# 4. 主程序
# ==========================================
def main():
    print("正在解析日志文件...")
    raw_data = parse_log(LOG_FILE)
    print(f"解析到 {len(raw_data)} 个系统")

    # 转换为长格式
    df_avg = compute_averages(raw_data)
    # 重塑为宽表：行=System, 列=(Model, TimeType)
    pivot = df_avg.pivot(index='System', columns='Model', values=['Total', 'Presolve', 'RootRelax', 'BB'])
    # 重新排序列顺序：先模型，再时间类型
    pivot = pivot.reorder_levels([1, 0], axis=1).sort_index(axis=1)

    # 保存
    pivot.to_csv(OUTPUT_CSV)
    print(f"汇总表已保存至 {OUTPUT_CSV}")
    print("\n表格预览（前几行）：")
    print(pivot.head())

if __name__ == "__main__":
    main()