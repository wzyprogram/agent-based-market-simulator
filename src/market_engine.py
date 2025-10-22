import numpy as np
import random
from dataclasses import dataclass, field
from typing import List, Dict

# =========================================================================================
# 1. Structures
# =========================================================================================

@dataclass
class Parameters:
    """ Three types of agents:
    - Low-frequency chartists: 
        "Trend followers", Buy/sell based on the deviation of prices from the moving average
    - Low-frequency fundamentalists: 
        "Fundamental trader", Trades in the opposite direction when the price deviates from the fundamental value
    - High-frequency traders:
        "Market makers: Randomly place limit orders or cancel orders based on LOB imbalance
    """
    N_Lt: int = 1       # number of chartists (low-freq)
    N_Lv: int = 1       # number of fundamentalists
    N_H: int = 8        # number of high-freq traders
    lambda_L: float = 20
    lambda_H: float = 15
    lambda_Lmin: float = 5
    lambda_Lmax: float = 40
    lambda_Hmin: float = 5
    lambda_Hmax: float = 40
    delta: float = 0.03
    kappa: float = 2.5
    nu: float = 2.0
    m0: float = 10000
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


# =========================================================================================
# 2. Initialize Agents
# =========================================================================================

@dataclass
class Chartist:
    ewma: float
    action_times: List[float]
    tau: float
    order_number: int = 0


@dataclass
class Fundamentalist:
    f_value: float
    action_times: List[float]


@dataclass
class HighFrequency:
    action_times: List[float]


def rexp(theta, theta_min, theta_max):
    """Truncated exponential sampling"""
    while True:
        x = np.random.exponential(theta)
        if theta_min <= x <= theta_max:
            return x


def agent_times(theta, theta_min, theta_max, T):
    """Generate action times in ms"""
    times = [0.0]
    counter = 0.0
    while counter < T:
        tau = rexp(theta, theta_min, theta_max) * 1000
        counter += tau
        times.append(counter)
    return times


def initialize_agents(params: Parameters):
    chartists, fundamentals, HFs = [], [], []

    for _ in range(params.N_Lt):
        at = agent_times(params.lambda_L, params.lambda_Lmin, params.lambda_Lmax, params.T)
        tau = np.mean(np.diff(at))
        chartists.append(Chartist(params.m0, at, tau))

    for _ in range(params.N_Lv):
        f_val = params.m0 * np.exp(np.random.normal(0, params.sigma))
        at = agent_times(params.lambda_L, params.lambda_Lmin, params.lambda_Lmax, params.T)
        fundamentals.append(Fundamentalist(f_val, at))

    for _ in range(params.N_H):
        at = agent_times(params.lambda_H, params.lambda_Hmin, params.lambda_Hmax, params.T)
        HFs.append(HighFrequency(at))

    return HFs, chartists, fundamentals


# =========================================================================================
# 3. Agent Rules
# =========================================================================================

def power_law(xm, alpha):
    """Draw volume from a power-law"""
    return xm / (np.random.rand() ** (1 / alpha))


def high_freq_action(order, lob: LOBState, params: Parameters):
    if order["type"] == "Limit":
        theta = lob.imbalance / 2 + 0.5
        order["side"] = "Sell" if np.random.rand() < theta else "Buy"
        if order["side"] == "Sell":
            alpha = 1 - (lob.imbalance / params.nu)
            eta = np.random.gamma(lob.spread, np.exp(lob.imbalance / params.kappa))
            order["price"] = int(lob.best_bid + 1 + eta)
        else:
            alpha = 1 + (lob.imbalance / params.nu)
            eta = np.random.gamma(lob.spread, np.exp(-lob.imbalance / params.kappa))
            order["price"] = int(lob.best_ask - 1 - eta)
        order["volume"] = int(power_law(10, alpha))
    return order


def chartist_action(order, lob: LOBState, chartist: Chartist, params: Parameters):
    if chartist.order_number + 1 >= len(chartist.action_times):
        return
    dt = chartist.action_times[chartist.order_number + 1] - chartist.action_times[chartist.order_number]
    lam = 1 - np.exp(-dt / chartist.tau)
    chartist.ewma += lam * (lob.midprice - chartist.ewma)
    chartist.order_number += 1

    x_m = 20
    if abs(lob.midprice - chartist.ewma) > (params.delta * lob.midprice):
        x_m = 50

    order["side"] = "Sell" if lob.midprice < chartist.ewma else "Buy"
    alpha = 1 - (lob.imbalance / params.nu) if order["side"] == "Sell" else 1 + (lob.imbalance / params.nu)
    order["volume"] = int(power_law(x_m, alpha))
    return order


def fundamentalist_action(order, lob: LOBState, fund: Fundamentalist, params: Parameters):
    x_m = 20
    if abs(lob.midprice - fund.f_value) > (params.delta * lob.midprice):
        x_m = 50
    order["side"] = "Sell" if fund.f_value < lob.midprice else "Buy"
    alpha = 1 - (lob.imbalance / params.nu) if order["side"] == "Sell" else 1 + (lob.imbalance / params.nu)
    order["volume"] = int(power_law(x_m, alpha))
    return order


# =========================================================================================
# 4. Update LOB State
# =========================================================================================

def update_lob(lob: LOBState, message: str):
    msg_parts = message.split("|")
    fields = msg_parts[0].split(",")
    msg_type, side, trader = fields[:3]
    executions = msg_parts[1:]

    for ex in executions:
        fid, price, vol = [int(x) for x in ex.split(",")]
        if msg_type == "New":
            if side == "Buy":
                lob.bids[fid] = LimitOrder(price, vol, trader)
            else:
                lob.asks[fid] = LimitOrder(price, vol, trader)
        elif msg_type == "Cancelled":
            (lob.bids if side == "Buy" else lob.asks).pop(fid, None)
        elif msg_type == "Trade":
            book = lob.asks if side == "Buy" else lob.bids
            if fid in book:
                book[fid].volume -= vol
                if book[fid].volume <= 0:
                    del book[fid]

    if lob.bids and lob.asks:
        lob.best_bid = max(o.price for o in lob.bids.values())
        lob.best_ask = min(o.price for o in lob.asks.values())
        bid_vol = sum(o.volume for o in lob.bids.values() if o.price == lob.best_bid)
        ask_vol = sum(o.volume for o in lob.asks.values() if o.price == lob.best_ask)
        lob.microprice = (lob.best_bid * bid_vol + lob.best_ask * ask_vol) / (bid_vol + ask_vol)
        total_bv = sum(o.volume for o in lob.bids.values())
        total_sv = sum(o.volume for o in lob.asks.values())
        lob.spread = abs(lob.best_ask - lob.best_bid)
        lob.midprice = (lob.best_ask + lob.best_bid) / 2
        lob.imbalance = (total_bv - total_sv) / (total_bv + total_sv) if (total_bv + total_sv) != 0 else 0.0


# =========================================================================================
# 5. Simulation (Simplified)
# =========================================================================================

def inject_simulation(params: Parameters, seed=1, steps=200):
    np.random.seed(seed)
    HFs, chartists, fundamentals = initialize_agents(params)

    lob = LOBState(
        spread=100, imbalance=0, midprice=params.m0, microprice=params.m0,
        price_ref=int(params.m0), best_bid=int(params.m0 - 50), best_ask=int(params.m0 + 50)
    )

    prices = [lob.midprice]

    for t in range(steps):
        # Randomly pick an agent and act
        agent_type = random.choice(["HF", "Chartist", "Fundamentalist"])
        order = {"type": "Limit", "side": None, "price": None, "volume": 0}

        if agent_type == "HF":
            order = high_freq_action(order, lob, params)
        elif agent_type == "Chartist":
            c = random.choice(chartists)
            order = chartist_action(order, lob, c, params)
        else:
            f = random.choice(fundamentals)
            order = fundamentalist_action(order, lob, f, params)

        # Update fake market price
        delta_p = random.uniform(-0.05, 0.05) * lob.midprice * 0.001
        lob.midprice += delta_p
        prices.append(lob.midprice)

    return np.array(prices)