"""
@Author:            ZHANG Biyuan
@Date:              2025/5/26
@Brief:             调度元件参数定义与组对称性精确识别方法
"""
from typing import List, Dict
from dataclasses import dataclass, field

@dataclass
class Thermal:
    """火电机组类"""
    # 机组运行参数
    pmax: float = 0.0
    pmin: float = 0.0
    RU: float = 0.0
    RD: float = 0.0
    UT: int = 0
    DT: int = 0
    cost_u: float = 0.0
    cost_d: float = 0.0
    u_max: int = 0
    d_max: int = 0
    ramp_fix_u: List[float] = field(default_factory=list)
    ramp_fix_d: List[float] = field(default_factory=list)

    # 成本函数参数
    bid_p: List[float] = field(default_factory=list)
    bid_pri: List[float] = field(default_factory=list)

    # 系统参数
    indices: List[int] = field(default_factory=list)
    buses: List[int] = field(default_factory=list)
    num: int = 1

    # 另一种成本形式
    a: float = 0.0
    b: float = 0.0
    c: float = 0.0

