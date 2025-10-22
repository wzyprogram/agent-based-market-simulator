import numpy as np
import random
from dataclasses import dataclass
from .structures import Parameters, LOBState
from utils.distributions import power_law, agent_times

# -------------------------------
# Agent classes
# -------------------------------

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


# -------------------------------
# Agent initialization
# -------------------------------

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


# -------------------------------
# Agent action functions
# -------------------------------

def high_freq_action(order, lob: LOBState, params: Parameters):
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
    """
    Logic - Look at the moving average: 
    Buy when the price is higher than the moving average and sell when it is lower
    """
    if chartist.order_number + 1 >= len(chartist.action_times):
        return
    dt = chartist.action_times[chartist.order_number + 1] - chartist.action_times[chartist.order_number]
    lam = 1 - np.exp(-dt / chartist.tau)
    chartist.ewma += lam * (lob.midprice - chartist.ewma)
    chartist.order_number += 1

    x_m = 20 # order size
    # Increase order size (if stronger signal)
    if abs(lob.midprice - chartist.ewma) > (params.delta * lob.midprice):
        x_m = 50

    order["side"] = "Sell" if lob.midprice < chartist.ewma else "Buy"
    # alpha: shape parameter for power-law
    alpha = 1 - (lob.imbalance / params.nu) if order["side"] == "Sell" else 1 + (lob.imbalance / params.nu)
    order["volume"] = int(power_law(x_m, alpha))
    return order


def fundamentalist_action(order, lob: LOBState, fund: Fundamentalist, params: Parameters):
    """
    Logic - Look at the fundamental value:
    Buy when the price is lower than the fundamental value and sell when it is higher
    """
    x_m = 20
    if abs(lob.midprice - fund.f_value) > (params.delta * lob.midprice):
        x_m = 50
    order["side"] = "Sell" if fund.f_value < lob.midprice else "Buy"
    alpha = 1 - (lob.imbalance / params.nu) if order["side"] == "Sell" else 1 + (lob.imbalance / params.nu)
    order["volume"] = int(power_law(x_m, alpha))
    return order