import numpy as np
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
    tick: float = 0.01
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

    bids: Dict[int, List[LimitOrder]] = field(default_factory=dict)
    asks: Dict[int, List[LimitOrder]] = field(default_factory=dict)


def initialize_realistic_lob(params, num_levels=5, base_depth=10):
    """
    Initialize a realistic queue-based LOB:
    - price levels contain LISTS of LimitOrder (FIFO)
    - Poisson-distributed depth with decay
    - realistic microprice, imbalance
    """
    mid = params.m0

    best_bid = int(mid - params.tick)
    best_ask = int(mid + params.tick)

    # Depth profile for each level
    depth_scale = base_depth * np.exp(-0.5 * np.arange(num_levels))
    bid_volumes = np.random.poisson(depth_scale)
    ask_volumes = np.random.poisson(depth_scale)

    # ensure top-of-book has volume
    bid_volumes[0] = max(bid_volumes[0], np.random.randint(5, 15))
    ask_volumes[0] = max(ask_volumes[0], np.random.randint(5, 15))

    bids = {}
    asks = {}

    # === BUILD FIFO QUEUES ===
    for i in range(num_levels):
        price_bid = best_bid - i * params.tick
        price_ask = best_ask + i * params.tick

        # Initialize queue (list!!)
        bids[price_bid] = [
            LimitOrder(price_bid, int(bid_volumes[i]), "system")
        ]
        asks[price_ask] = [
            LimitOrder(price_ask, int(ask_volumes[i]), "system")
        ]

    # === Compute imbalance ===
    total_bid = bid_volumes.sum()
    total_ask = ask_volumes.sum()
    imbalance = (total_bid - total_ask) / (total_bid + total_ask)

    # === Queue-based microprice ===
    best_bid_vol = bid_volumes[0]
    best_ask_vol = ask_volumes[0]
    microprice = (best_ask * best_bid_vol + best_bid * best_ask_vol) / (
        best_bid_vol + best_ask_vol
    )

    return LOBState(
        spread = best_ask - best_bid,
        imbalance = float(imbalance),
        midprice = mid,
        microprice = microprice,
    price_ref = int(mid),
        best_bid = best_bid,
        best_ask = best_ask,
        bids = bids,   # Dict[int, List[LimitOrder]]
        asks = asks
    )