import random
import numpy as np
from .structures import Parameters, LOBState
from .agents import initialize_agents, high_freq_action, chartist_action, fundamentalist_action
from .lob import update_lob


def inject_simulation(params: Parameters, seed=1, steps=200):
    np.random.seed(seed)
    random.seed(seed)

    # === Initialize ===
    HFs, chartists, fundamentals = initialize_agents(params)
    lob = LOBState(
        spread=100, imbalance=0, midprice=params.m0, microprice=params.m0,
        price_ref=int(params.m0), best_bid=int(params.m0 - 50), best_ask=int(params.m0 + 50)
    )

    prices = [lob.midprice]

    # === Simulation loop ===
    for t in range(steps):
        agent_type = random.choice(["HF", "Chartist", "Fundamentalist"])
        order = {"type": "Limit", "side": None, "price": None, "volume": 0, "trader": agent_type}

        # Agents decide action
        if agent_type == "HF":
            order = high_freq_action(order, lob, params)
        elif agent_type == "Chartist":
            c = random.choice(chartists)
            order = chartist_action(order, lob, c, params)
        else:
            f = random.choice(fundamentals)
            order = fundamentalist_action(order, lob, f, params)

        # === Convert the order to a market message ===
        price = order["price"] if order["price"] is not None else int(lob.midprice)
        msg = f"New,{order['side']},{order['trader']}|1,{price},{order['volume']}"
        update_lob(lob, msg)
        update_lob(lob, msg)

        # === Record new market state ===
        prices.append(lob.midprice)

    return np.array(prices)