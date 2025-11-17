from dataclasses import dataclass, field
from typing import Dict, List
from .structures import LOBState, LimitOrder


class QueueLOBEngine:
    """
    A realistic queue-based Limit Order Book engine.
    - Each price level contains a FIFO queue of LimitOrders
    - Supports limit orders, market orders, and cancellations
    - Automatically updates best bid, best ask, midprice, microprice, imbalance
    """

    def __init__(self, lob: LOBState):
        self.lob = lob

    # -------------------------------------------------------------
    # Submit LIMIT ORDER
    # -------------------------------------------------------------
    def submit_limit(self, side: str, price: int, volume: int, trader: str):
        """
        Add an order to the bid/ask queue.
        """
        book = self.lob.bids if side == "Buy" else self.lob.asks

        if price not in book:
            book[price] = []

        MAX_QUEUE_PER_LEVEL = 10
        # prevent infinite LOB growth
        if len(book[price]) > MAX_QUEUE_PER_LEVEL:
            book[price].pop(0)  # remove oldest

        # append to queue (FIFO)
        book[price].append(LimitOrder(price, volume, trader))
        self._update_state()
        return []

    # -------------------------------------------------------------
    # Submit MARKET ORDER
    # -------------------------------------------------------------
    def submit_market(self, side: str, volume: int, trader: str):
        """
        Market order consumes opposite side queues.
        """
        trades = []
        book = self.lob.asks if side == "Buy" else self.lob.bids

        qty = volume

        # sorted price levels
        price_levels = sorted(book.keys()) if side == "Buy" else sorted(book.keys(), reverse=True)

        for px in price_levels:
            queue = book[px]

            while queue and qty > 0:
                head_order = queue[0]
                trade_size = min(qty, head_order.volume)

                # record trade
                trades.append((side, px, trade_size, trader, head_order.trader))

                # update remaining quantities
                head_order.volume -= trade_size
                qty -= trade_size

                if head_order.volume == 0:
                    queue.pop(0)  # FIFO: remove head
                if not queue:
                    del book[px]
                    break

            if qty == 0:
                break

        self._update_state()
        return trades

    # -------------------------------------------------------------
    # Cancel an order at a given level 
    # -------------------------------------------------------------
    def cancel_level(self, side: str, price: int):
        """
        Cancels one entire price level (simple version).
        """
        book = self.lob.bids if side == "Buy" else self.lob.asks
        if price in book:
            del book[price]
        self._update_state()

    # -------------------------------------------------------------
    # INTERNAL: Recompute best bid/ask etc.
    # -------------------------------------------------------------
    def _update_state(self):
        bids = self.lob.bids
        asks = self.lob.asks

        # -------- best bid --------
        if bids:
            self.lob.best_bid = max(bids.keys())
        else:
            self.lob.best_bid = None

        # -------- best ask --------
        if asks:
            self.lob.best_ask = min(asks.keys())
        else:
            self.lob.best_ask = None

        # -------- midprice --------
        if self.lob.best_bid is not None and self.lob.best_ask is not None:
            self.lob.midprice = (self.lob.best_bid + self.lob.best_ask) / 2
            self.lob.spread  = self.lob.best_ask - self.lob.best_bid
        elif self.lob.best_bid is not None:
            # only bids left → midprice follows best_bid
            self.lob.midprice = self.lob.best_bid
            self.lob.spread = None
        elif self.lob.best_ask is not None:
            # only asks left → midprice follows best_ask
            self.lob.midprice = self.lob.best_ask
            self.lob.spread = None
        else:
            # orderbook entirely empty (rare)
            self.lob.midprice = None
            self.lob.spread = None

        # -------- microprice --------
        if (
            self.lob.best_bid is not None
            and self.lob.best_ask is not None
            and len(bids[self.lob.best_bid]) > 0
            and len(asks[self.lob.best_ask]) > 0
        ):
            bid_vol = sum(o.volume for o in bids[self.lob.best_bid])
            ask_vol = sum(o.volume for o in asks[self.lob.best_ask])
            self.lob.microprice = (
                self.lob.best_bid * ask_vol + self.lob.best_ask * bid_vol
            ) / (bid_vol + ask_vol)
        else:
            self.lob.microprice = self.lob.midprice   # fallback

        # -------- imbalance --------
        total_bid_vol = sum(o.volume for q in bids.values() for o in q)
        total_ask_vol = sum(o.volume for q in asks.values() for o in q)

        if total_bid_vol + total_ask_vol > 0:
            self.lob.imbalance = (total_bid_vol - total_ask_vol) / (
                total_bid_vol + total_ask_vol
            )
        else:
            self.lob.imbalance = 0.0