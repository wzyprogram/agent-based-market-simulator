import random
import numpy as np
from .structures import Parameters, LOBState, initialize_realistic_lob
from .agents import initialize_agents, high_freq_action, chartist_action, fundamentalist_action
from .lob import update_lob
from .queue import QueueLOBEngine
from copy import deepcopy

def inject_simulation(params: Parameters, seed=1, steps=200):
    np.random.seed(seed)
    random.seed(seed)

    # === Initialize ===
    HFs, chartists, fundamentals = initialize_agents(params)
    engine = QueueLOBEngine(initialize_realistic_lob(params))

    midprices = [engine.lob.midprice]
    best_bids = [engine.lob.best_bid]
    best_asks = [engine.lob.best_ask]

    # history for heatmap (store deep copy of queues)
    lob_history = [(deepcopy(engine.lob.bids), deepcopy(engine.lob.asks))]

    # === Simulation loop ===
    for t in range(steps):
        agent_type = random.choice(["HF", "Chartist", "Fundamentalist"])
        # order = {"type": "Limit", "side": None, "price": None, "volume": 0, "trader": agent_type}
        order_type = random.choices(
            ["Limit", "Market", "Cancel"],
            weights=[0.6, 0.3, 0.1],    
            # 60% limit (add liquidity), 30% market (consume liquidity), 10% cancel (remove stale orders)
            k=1
        )[0]
        order = {"type": order_type, "side": None, "price": None, "volume": 0, "trader": agent_type}

        # Agents decide action
        if agent_type == "HF":
            order = high_freq_action(order, engine.lob, params)
        elif agent_type == "Chartist":
            c = random.choice(chartists)
            order = chartist_action(order, engine.lob, c, params)
        else:
            f = random.choice(fundamentals)
            order = fundamentalist_action(order, engine.lob, f, params)

        # pure agent-based order
        # price = order["price"] if order["price"] is not None else int(lob.midprice)
        # msg = f"New,{order['side']},{order['trader']}|1,{price},{order['volume']}"
        # update_lob(lob, msg)

        # add queue-based precessing    
        if order["type"] == "Limit":
            engine.submit_limit(
                order["side"],
                order["price"],
                order["volume"],
                order["trader"]
            )

        elif order["type"] == "Market":
            engine.submit_market(
                order["side"],
                order["volume"],
                order["trader"]
            )
        
        else:  # Cancel
            engine.cancel_level(
                order["side"],
                order["price"]
            )

        # === Record new market state ===
        midprices.append(engine.lob.midprice)
        best_bids.append(engine.lob.best_bid)
        best_asks.append(engine.lob.best_ask)

        lob_history.append((deepcopy(engine.lob.bids), deepcopy(engine.lob.asks)))

    # return everything needed
    return {
        "midprices": np.array(midprices),
        "best_bids": best_bids,
        "best_asks": best_asks,
        "lob_history": lob_history,
        "final_lob": engine.lob
    }