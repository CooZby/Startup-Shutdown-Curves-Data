"""
@Author:            ZHANG Biyuan
@Date:              2025/5/26
@Brief:             电力系统参数读取
"""
from DataDef import Thermal
import pandas as pd
from typing import List, Dict
import numpy as np


class DataReader:
    """数据读取"""
    def __init__(self, data_path: str, l_rate=1.0):
        self.data_path = "inputdata/" + data_path
        self.data_file = pd.ExcelFile(self.data_path)
        self.l_rate = l_rate
        self.random_seed = 42

        # 主要参数
        self.DayMax = 20
        self.seg = 5
        self.duration: int = 96
        self.thermals: List[Thermal] = []
        self.slf = []
        self.load_offset = {}
        self.loadc = {}

        self._parse_all_sheets()
        # self._apply_symmetry_detection()

    def _parse_all_sheets(self):
        # 1. 解析基础参数 sheet 'para'
        self._parse_para_value()

        # 2. 解析负荷曲线 sheet 'loadc'
        self._parse_slf_generators()

        # 3. 解析线路及 PTDF sheet 'bra'
        self._parse_bra_ptdf()

        # 4. 解析火电机组 sheet 'thermal'
        self._parse_thermal_basic()

        # 其他表单可继续添加

    def _parse_thermal_basic(self):
        """解析火电机组参数及报价、爬坡等参数"""
        from func import quad_cost, split_power_segments, generate_ramp_fix

        thermal_df = pd.read_excel(self.data_file, sheet_name="thermal")
        derta_t = self.duration / self.input_duration

        self.thermals = []
        idx_map = {}

        for idx, row in thermal_df.iterrows():

            thermal = Thermal()

            # ======================
            # 机组序号处理
            # ======================
            if "机组序号" in thermal_df.columns:
                gid = int(row["机组序号"])
            else:
                gid = idx + 1
            thermal.indices = [gid]

            # ======================
            # 基本参数
            # ======================
            pmax = float(row.get("pmax", 0))
            pmin = float(row.get("pmin", 0))
            thermal.pmax = pmax
            thermal.pmin = pmin

            r = float(row.get("r", 0))
            thermal.RU = r / derta_t
            thermal.RD = r / derta_t

            thermal.UT = int(row.get("ton", 0) * derta_t)
            thermal.DT = int(row.get("toff", 0) * derta_t)

            # u_max & d_max 可能缺失
            if "u_max" in thermal_df.columns:
                thermal.u_max = int(row.get("u_max", 0))
            else:
                # IF(pmax>300,3,IF(pmax>=100,4,5))
                if pmax > 300:
                    thermal.u_max = 3
                elif pmax >= 100:
                    thermal.u_max = 4
                else:
                    thermal.u_max = 5

            if "d_max" in thermal_df.columns:
                thermal.d_max = int(row.get("d_max", 0))
            else:
                thermal.d_max = thermal.u_max

            # ======================
            # 成本参数
            # ======================
            thermal.cost_u = float(row.get("fixed", 0))
            thermal.cost_d = float(row.get("fixed", 0))

            thermal.a = float(row.get("a", 0))
            thermal.b = float(row.get("b", 0))
            thermal.c = float(row.get("c", 0))

            # ======================
            # 挂接母线
            # ======================
            thermal.buses = [int(row.get("busno", 0))]

            # ======================
            # === 分段功率 bid_p ===
            # ======================
            seg = self.seg
            bid_p = split_power_segments(pmin, pmax, seg)
            thermal.bid_p = bid_p

            # ======================
            # === 分段价格 bid_pri ===
            # 对应 self.seg + 1 个点：pmin → pmax
            # ======================
            points = [pmin] + [pmin + sum(bid_p[:i]) for i in range(1, len(bid_p) + 1)]
            thermal.bid_pri = [quad_cost(p, thermal.a, thermal.b, thermal.c) for p in points]

            # ======================
            # === ramp_fix_u / ramp_fix_d ===
            # ======================
            # 不同 duration 对应不同长度分组
            if self.duration == 96:
                if pmax < 50:
                    length = 1
                elif pmax < 150:
                    length = 3
                elif pmax < 200:
                    length = 3
                elif pmax < 300:
                    length = 4
                elif pmax < 600:
                    length = 6
                else:
                    length = 8

            elif self.duration == 24:
                if pmax < 100:
                    length = 1
                elif pmax < 150:
                    length = 2
                elif pmax < 200:
                    length = 2
                elif pmax < 300:
                    length = 2
                elif pmax < 600:
                    length = 3
                else:
                    length = 4
            else:
                raise ValueError("未定义该 duration 对应的 ramp_fix 长度规则")

            fix_u, fix_d = generate_ramp_fix(length, pmin)
            thermal.ramp_fix_u = fix_u
            thermal.ramp_fix_d = fix_d
            thermal.UT = thermal.UT + len(fix_d) + len(fix_u)

            # 保存
            self.thermals.append(thermal)
            idx_map[gid] = len(self.thermals) - 1

    def _parse_para_value(self):
        """解析基本参数：把第一列作为索引读取参数值。

        要求 sheet 'para' 第一列为参数名（例如 'duration','busno' ...），
        第二列为对应的值。函数对列名不做强依赖，自动取第一列数据。
        """
        # 读表并把第一列设为 index（参数名）
        try:
            para_df = pd.read_excel(self.data_file, sheet_name="para", index_col=0)
        except Exception as e:
            raise RuntimeError(f"读取参数文件失败: {e}")

        if para_df.shape[1] == 0:
            raise ValueError("sheet 'para' 中没有数据列，确认第一列为参数名，第二列为值。")

        # 取第一个数据列名（有时此列名可能是 'value' 或其他）
        data_col = para_df.columns[0]

        def _get(key, cast_type, default=None):
            """安全读取 index 为 key 的值并转换类型"""
            try:
                if key not in para_df.index:
                    if default is not None:
                        return default
                    raise KeyError(f"参数 '{key}' 在表中未找到。")
                val = para_df.at[key, data_col]
                if pd.isna(val):
                    if default is not None:
                        return default
                    raise ValueError(f"参数 '{key}' 的值为空。")
                return cast_type(val)
            except Exception as e:
                raise type(e)(f"读取参数 '{key}' 失败: {e}")

        # 读取并赋值（按你原来的字段）
        self.input_duration = _get("duration", int)
        self.input_busno = _get("busno", int)
        self.input_brano = _get("brano", int)
        self.input_swing = _get("swing", int)
        self.input_reserve = _get("reserve", float)

    def _parse_slf_generators(self, fig_flag=False):
        """解析负荷曲线"""
        from func import interpolate_day_load, plot_all_load_curves

        slf_df = pd.read_excel(self.data_file, sheet_name="loadc")

        for i in range(self.DayMax):
            temp_loadc = list(
                slf_df.iloc[i * self.input_duration: (i + 1) * self.input_duration]["c1"]
            )

            if self.input_duration != self.duration:
                interpolated = interpolate_day_load(temp_loadc, self.duration, method="cubic")
                self.loadc[f"day_{i + 1}"] = list(interpolated)
            else:
                self.loadc[f"day_{i + 1}"] = temp_loadc


        # ------------------------
        # 读取 dem 表单并求和
        # ------------------------
        load_df = pd.read_excel(self.data_file, sheet_name="load")

        # 求 rt 列数据之和
        self.load_sum = load_df["rt"].sum()

        if fig_flag:
            plot_all_load_curves(self.loadc)

    def _parse_bra_ptdf(self):
        """生成线路 PTDF 字典 & 计算负荷对线路的偏移（load_offset）"""
        # ==============================
        # Step 1: 读取完整数据
        # ==============================
        bra_df = pd.read_excel(self.data_file, sheet_name="bra")
        brano_full = len(bra_df)
        busno = self.input_busno
        swing = self.input_swing - 1  # 0-based

        # ==============================
        # Step 2: 用完整网络构建 B 矩阵
        # ==============================
        B = np.zeros((busno, busno))
        for i in range(brano_full):
            b1 = int(bra_df.loc[i, "b1"]) - 1
            b2 = int(bra_df.loc[i, "b2"]) - 1
            x = float(bra_df.loc[i, "x"])
            invx = 1 / x
            B[b1, b2] -= invx
            B[b2, b1] -= invx
            B[b1, b1] += invx
            B[b2, b2] += invx

        # ==============================
        # Step 3: 计算全网阻抗矩阵 X
        # ==============================
        B_mod = B.copy()
        B_mod[swing, :] = float(1E+8)
        B_mod[:, swing] = float(1E+8)
        X = np.linalg.inv(B_mod)

        # ==============================
        # Step 4: 构建全网 BL 和 A
        # ==============================
        BL_full = np.zeros((brano_full, brano_full))
        A_full = np.zeros((brano_full, busno))
        for i in range(brano_full):
            BL_full[i, i] = 1 / bra_df.loc[i, "x"]
            A_full[i, int(bra_df.loc[i, "b1"]) - 1] = 1
            A_full[i, int(bra_df.loc[i, "b2"]) - 1] = -1

        # ==============================
        # Step 5: 计算全网 PTDF（所有线路）
        # ==============================
        PTDF_full = BL_full @ A_full @ X
        PTDF_full[np.abs(PTDF_full) <= 1e-5] = 0

        # ==============================
        # Step 6: 根据 l_rate 随机选择线路子集（✅ 在 PTDF 之后！）
        # ==============================
        n_selected = max(1, int(np.ceil(brano_full * self.l_rate)))

        if n_selected >= brano_full:
            selected_indices = np.arange(brano_full)
        else:
            # 🔑 随机种子应在更高层设置（如 __init__），此处不设！
            # 如果必须在此控制，可用类属性 self.random_seed
            if hasattr(self, 'random_seed'):
                np.random.seed(self.random_seed)
            selected_indices = np.random.choice(
                brano_full, size=n_selected, replace=False
            )
            selected_indices = np.sort(selected_indices)  # 保持顺序（可选）

        # 提取子集
        bra_df_sub = bra_df.iloc[selected_indices].reset_index(drop=True)
        PTDF_sub = PTDF_full[selected_indices, :]
        brano = len(selected_indices)

        # ==============================
        # Step 7: 读取负荷
        # ==============================
        load_df = pd.read_excel(self.data_file, sheet_name="load")
        bus_load = {int(row.busno): float(row.rt) for _, row in load_df.iterrows()}

        # ==============================
        # Step 8: 构建 bra_dict（仅子集）
        # ==============================
        self.bra_dict = {}
        self.load_offset_Mar = {f"day_{d + 1}": {} for d in range(self.DayMax)}

        for l in range(brano):
            orig_idx = selected_indices[l]  # 原始索引
            row = bra_df.iloc[orig_idx]

            bus1 = int(row["b1"])
            bus2 = int(row["b2"])
            s = float(row["s"])
            state = int(row["state"]) if "state" in bra_df.columns else 1

            ptdf_dict = {bus + 1: float(PTDF_sub[l, bus]) for bus in range(busno)}

            self.bra_dict[l + 1] = {
                "b1": bus1,
                "b2": bus2,
                "s": s,
                "state": state,
                "ptdf": ptdf_dict,
                "original_id": int(orig_idx + 1)  # 原始线路编号（1-based）
            }

            load_offset_sum = sum(
                ptdf_dict.get(bus, 0) * bus_load.get(bus, 0)
                for bus in range(1, busno + 1)
            )
            for d in range(self.DayMax):
                self.load_offset_Mar[f"day_{d + 1}"][l + 1] = [
                    load_offset_sum * self.loadc[f"day_{d + 1}"][t]
                    for t in range(self.duration)
                ]


class DataReader_GWCpt:
    """数据读取"""
    def __init__(self, data_path: str):
        self.data_path = "inputdata/" + data_path
        self.data_file = pd.ExcelFile(self.data_path)

        self.duration: int = 0
        self.thermals: List[Thermal] = []
        self.slf: List[float] = []

        self._parse_all_sheets()
        # self._apply_symmetry_detection()

    def _parse_all_sheets(self):
        for sheet_name in self.data_file.sheet_names:
            if sheet_name == "slf":
                self._parse_slf_generators()
            elif sheet_name == "thermal":
                self._parse_thermal_basic()
            # 其他表单可继续添加

    def _parse_thermal_basic(self):
        """解析火电机组参数及报价、爬坡等参数"""
        # 读取所有相关表
        thermal_df = pd.read_excel(self.data_file, sheet_name="thermal")
        bid_p_df = pd.read_excel(self.data_file, sheet_name="bid_p")
        bid_pri_df = pd.read_excel(self.data_file, sheet_name="bid_price")
        ramp_fix_df = pd.read_excel(self.data_file, sheet_name="ramp_fix")

        # 按机组序号建立索引映射，方便后续查找
        idx_map = {}
        self.thermals = []
        for idx, row in thermal_df.iterrows():
            thermal = Thermal()
            # 基本参数
            thermal.indices = [int(row.get('机组序号', 0))]
            thermal.pmax = float(row.get('pmax', 0))
            thermal.pmin = float(row.get('pmin', 0))
            thermal.RU = float(row.get('RU', 0))
            thermal.RD = float(row.get('RD', 0))
            thermal.UT = int(row.get('UT', 0))
            thermal.DT = int(row.get('DT', 0))
            thermal.u_max = int(row.get('u_max', 0))
            thermal.d_max = int(row.get('d_max', 0))
            thermal.cost_u = float(row.get('cost_u', 0))
            thermal.cost_d = float(row.get('cost_d', 0))
            # bus参数
            thermal.buses = [int(row.get('busno', 0))]

            self.thermals.append(thermal)
            idx_map[thermal.indices[0]] = len(self.thermals) - 1  # 机组序号到对象的映射

        # 成本容量
        for _, row in bid_p_df.iterrows():
            unit_no = int(row.get('机组序号', 0))
            if unit_no in idx_map:
                thermal = self.thermals[idx_map[unit_no]]
                # 读取 bid_p 数据，并忽略空值
                bid_p_values = [float(x) for x in row.values[1:] if pd.notnull(x)]
                # # 在开头插入 pmin
                # bid_p_values.insert(0, thermal.pmin)
                # 赋值给 bid_p
                thermal.bid_p = bid_p_values

        # 成本价格
        for _, row in bid_pri_df.iterrows():
            unit_no = int(row.get('机组序号', 0))
            if unit_no in idx_map:
                self.thermals[idx_map[unit_no]].bid_pri = [
                    float(x) for x in row.values[1:] if pd.notnull(x)
                ]

        # ramp_fix: 机组启动/停机过程
        ramp_fix_columns = list(ramp_fix_df.columns)
        try:
            start_idx = next(i for i, col in enumerate(ramp_fix_columns) if "起始数据" in str(col))
        except StopIteration:
            raise ValueError("ramp_fix表格未找到'起始数据'列")

        for _, row in ramp_fix_df.iterrows():
            unit_no = int(row.get('机组序号', 0))
            if unit_no in idx_map:
                # 只取从“起始数据”这一列开始的所有数据
                values = [float(x) for x in row.values[start_idx:] if pd.notnull(x)]
                thermal = self.thermals[idx_map[unit_no]]
                if len(values) == 0:
                    continue
                pmin = thermal.pmin
                try:
                    # 找到第一个大于等于pmin的索引
                    up_end = next(i for i, v in enumerate(values) if v >= pmin)
                    thermal.ramp_fix_u = values[:up_end + 1]
                    thermal.ramp_fix_d = values[up_end + 1:]
                    thermal.UT += len(thermal.ramp_fix_u) + len(thermal.ramp_fix_d)
                except StopIteration:
                    thermal.ramp_fix_u = values
                    thermal.ramp_fix_d = []

    def _parse_slf_generators(self):
        """解析负荷曲线"""
        slf_df = pd.read_excel(self.data_file, sheet_name="slf")
        self.slf = [float(row.get('系统负荷大小（MW）', 0)) for _, row in slf_df.iterrows()]
        self.duration = len(self.slf)

