import numpy as np
import random
from dataclasses import dataclass

from .structures import Parameters, LOBState
from utils.distributions import power_law, agent_times


# ============================================================
# Agent data classes
# ============================================================

@dataclass
class Chartist:
    ewma: float
    action_times: list
    tau: float
    order_number: int = 0


@dataclass
class Fundamentalist:
    f_value: float
    action_times: list


@dataclass
class HighFrequency:
    action_times: list


# ============================================================
# Agent initialization
# ============================================================

def initialize_agents(params: Parameters):
    chartists, fundamentals, HFs = [], [], []

    # Chartists
    for _ in range(params.N_Lt):
        at = agent_times(params.lambda_L,
                         params.lambda_Lmin,
                         params.lambda_Lmax,
                         params.T)
        tau = np.mean(np.diff(at))
        chartists.append(Chartist(params.m0, at, tau))

    # Fundamentalists
    for _ in range(params.N_Lv):
        f_val = params.m0 * np.exp(np.random.normal(0, params.sigma))
        at = agent_times(params.lambda_L,
                         params.lambda_Lmin,
                         params.lambda_Lmax,
                         params.T)
        fundamentals.append(Fundamentalist(f_val, at))

    # High-Frequency traders
    for _ in range(params.N_H):
        at = agent_times(params.lambda_H,
                         params.lambda_Hmin,
                         params.lambda_Hmax,
                         params.T)
        HFs.append(HighFrequency(at))

    return HFs, chartists, fundamentals


# ============================================================
# High-frequency trader behavior
# ============================================================

def high_freq_action(order, lob: LOBState, params: Parameters):
    """
    High-frequency traders react to order flow imbalance.
    Small volume, aggressive positioning around the best bid/ask.
    """
    tick = params.tick
    best_bid = lob.best_bid if lob.best_bid is not None else lob.price_ref - tick
    best_ask = lob.best_ask if lob.best_ask is not None else lob.price_ref + tick

    # Side decision based on imbalance
    theta = lob.imbalance / 2 + 0.5
    order["side"] = "Sell" if np.random.rand() < theta else "Buy"

    # -------------------- CANCEL --------------------
    if order["type"] == "Cancel":
        book = lob.bids if order["side"] == "Buy" else lob.asks
        if not book:
            order["type"] = "Limit"
        else:
            order["price"] = random.choice(list(book.keys()))
            order["volume"] = 0
            return order

    # -------------------- MARKET --------------------
    if order["type"] == "Market":
        order["volume"] = np.random.randint(1, 4)
        return order

    # -------------------- LIMIT --------------------
    spread = lob.spread if (lob.spread is not None and lob.spread > 0) else tick

    if order["side"] == "Sell":
        eta = np.random.gamma(spread, np.exp(lob.imbalance / params.kappa))
        price = best_bid + tick + eta
    else:
        eta = np.random.gamma(spread, np.exp(-lob.imbalance / params.kappa))
        price = best_ask - tick - eta

    order["price"] = float(price)

    # Volume: small but responsive to imbalance
    base_volume = np.random.randint(1, 4)
    scale = 1 + 0.5 * abs(lob.imbalance)
    vol = int(base_volume * scale)
    order["volume"] = max(1, min(vol, 8))

    return order


# ============================================================
# Chartist behavior
# ============================================================

def chartist_action(order, lob: LOBState, chartist: Chartist, params: Parameters):
    """
    Chartists follow short-term price trends.
    They compare the midprice to their moving average (EWMA).
    """
    tick = params.tick
    best_bid = lob.best_bid if lob.best_bid is not None else lob.price_ref - tick
    best_ask = lob.best_ask if lob.best_ask is not None else lob.price_ref + tick

    # Update EWMA
    if chartist.order_number + 1 >= len(chartist.action_times):
        return order

    dt = chartist.action_times[chartist.order_number + 1] - \
         chartist.action_times[chartist.order_number]
    lam = 1 - np.exp(-dt / chartist.tau)
    chartist.ewma += lam * (lob.midprice - chartist.ewma)
    chartist.order_number += 1

    # Side decision: follow trend
    order["side"] = "Buy" if lob.midprice > chartist.ewma else "Sell"

    # -------------------- CANCEL --------------------
    if order["type"] == "Cancel":
        book = lob.bids if order["side"] == "Buy" else lob.asks
        if not book:
            order["type"] = "Limit"
        else:
            order["price"] = random.choice(list(book.keys()))
            order["volume"] = 0
            return order

    # -------------------- MARKET --------------------
    if order["type"] == "Market":
        order["volume"] = np.random.randint(1, 4)
        return order

    # -------------------- LIMIT --------------------
    signal_strength = abs(lob.midprice - chartist.ewma)
    aggressive = signal_strength > params.delta * lob.midprice

    if order["side"] == "Buy":
        price = best_ask if aggressive else best_bid - tick
    else:
        price = best_bid if aggressive else best_ask + tick

    order["price"] = float(price)

    # Volume scaling with trend strength
    base_volume = np.random.randint(3, 8)
    scale = 1 + min(signal_strength / (0.01 * lob.midprice), 2.0)

    vol = int(base_volume * scale)
    order["volume"] = max(1, min(vol, 12))

    return order


# ============================================================
# Fundamentalist behavior
# ============================================================

def fundamentalist_action(order, lob: LOBState, fund: Fundamentalist, params: Parameters):
    """
    Fundamentalists trade toward the fundamental value.
    Larger deviation → stronger/more aggressive trades.
    """
    tick = params.tick
    best_bid = lob.best_bid if lob.best_bid is not None else lob.price_ref - tick
    best_ask = lob.best_ask if lob.best_ask is not None else lob.price_ref + tick

    # Side: revert to fundamental value
    order["side"] = "Buy" if lob.midprice < fund.f_value else "Sell"

    deviation = abs(lob.midprice - fund.f_value)
    aggressive = deviation > params.delta * lob.midprice

    # -------------------- CANCEL --------------------
    if order["type"] == "Cancel":
        book = lob.bids if order["side"] == "Buy" else lob.asks
        if not book:
            order["type"] = "Limit"
        else:
            order["price"] = random.choice(list(book.keys()))
            order["volume"] = 0
            return order

    # -------------------- MARKET --------------------
    if order["type"] == "Market":
        base_volume = np.random.randint(2, 5)
        scale = 1 + min(deviation / (0.02 * lob.midprice), 3.0)
        vol = int(base_volume * scale)
        order["volume"] = max(2, min(vol, 15))
        return order

    # -------------------- LIMIT --------------------
    if order["side"] == "Buy":
        price = best_bid - tick if not aggressive else best_ask
    else:
        price = best_ask + tick if not aggressive else best_bid

    order["price"] = float(price)

    # Volume depends on mispricing magnitude
    base_volume = np.random.randint(5, 15)
    misprice = abs(fund.f_value - lob.midprice) / lob.midprice
    scale = 1 + min(misprice * 10, 3.0)

    vol = int(base_volume * scale)
    order["volume"] = max(3, min(vol, 20))

    return order