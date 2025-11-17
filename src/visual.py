import matplotlib.pyplot as plt
import numpy as np

def plot_l1(midprices, best_bids, best_asks):
    
    plt.figure(figsize=(10,5))
    plt.plot(best_bids, label="Best Bid", color="blue", alpha=0.7)
    plt.plot(best_asks, label="Best Ask", color="red", alpha=0.7)
    plt.plot(midprices,   label="Midprice", color="black", linewidth=2)

    plt.title("Top-of-Book Evolution")
    plt.xlabel("Time Step")
    plt.ylabel("Price")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.show()

def plot_heatmap(lob_history):

    prices = sorted(set(
        px for b,a in lob_history for px in list(b.keys()) + list(a.keys())
    ))
    price_to_idx = {p:i for i,p in enumerate(prices)}

    depth_matrix = np.zeros((len(prices), len(lob_history)))

    for t, (bids, asks) in enumerate(lob_history):
        for px, queue in bids.items():
            depth_matrix[price_to_idx[px], t] += sum(o.volume for o in queue)
        for px, queue in asks.items():
            depth_matrix[price_to_idx[px], t] -= sum(o.volume for o in queue)

    plt.figure(figsize=(10,7))
    plt.imshow(depth_matrix, cmap="bwr", aspect="auto", origin="lower")
    plt.colorbar(label="Bid Vol (pos) / Ask Vol (neg)")
    plt.title("Order Book Depth Heatmap")
    plt.xlabel("Time")
    plt.ylabel("Price Level")
    plt.show()

def plot_depth_curve(lob):
    import matplotlib.pyplot as plt
    
    bid_prices = sorted(lob.bids.keys(), reverse=True)
    ask_prices = sorted(lob.asks.keys())

    bid_depth = [sum(o.volume for o in lob.bids[p]) for p in bid_prices]
    ask_depth = [sum(o.volume for o in lob.asks[p]) for p in ask_prices]

    plt.figure(figsize=(8,5))
    plt.step(bid_prices, bid_depth, label="Bid Depth", color="blue", where="mid")
    plt.step(ask_prices, ask_depth, label="Ask Depth", color="red", where="mid")

    plt.title("LOB Depth Curve (Snapshot)")
    plt.xlabel("Price")
    plt.ylabel("Depth")
    plt.grid(alpha=0.3)
    plt.legend()
    plt.show()