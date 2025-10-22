from .structures import LOBState, LimitOrder

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