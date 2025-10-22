from dataclasses import dataclass, field
from typing import List, Dict

@dataclass
class Parameters:
    """Global simulation parameters"""
    N_Lt: int = 1
    N_Lv: int = 1
    N_H: int = 8
    lambda_L: float = 20
    lambda_H: float = 15
    lambda_Lmin: float = 5
    lambda_Lmax: float = 40
    lambda_Hmin: float = 5
    lambda_Hmax: float = 40
    delta: float = 0.03
    kappa: float = 2.5
    nu: float = 2.0
    m0: float = 100
    sigma: float = 0.1
    T: float = 3600 * 1000  # milliseconds


@dataclass
class LimitOrder:
    price: int
    volume: int
    trader: str


@dataclass
class MovingAverage:
    mean_price: float
    action_times: List[float]
    tau: float


@dataclass
class LOBState:
    spread: int
    imbalance: float
    midprice: float
    microprice: float
    price_ref: int
    best_bid: int
    best_ask: int
    bids: Dict[int, LimitOrder] = field(default_factory=dict)
    asks: Dict[int, LimitOrder] = field(default_factory=dict)