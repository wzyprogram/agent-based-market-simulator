# CSE6730 2025 Fall Group 12 Final Project
This repository contains our group project materials for CSE6730 (Modeling and Simulation), Fall 2025.  
Includes folders for:
- data/
- src/
- report/
- literature/
- checkpoint/

## Market Simulation Framework

This project develops an event-driven market simulation that integrates three key components inspired by real-world market microstructure:

- **Queue-Based Matching Engine:** A full FIFO price-time priority engine tracks individual orders at each price level, handles marketable limit orders, and updates best bid/ask quotes in real time.

- **Heterogeneous Trading Agents:** Fundamentalists, Chartists, and High-Frequency Traders act asynchronously according to independent Poisson processes. Their trading decisions depend on mispricing, momentum, or inventory conditions, generating continuous liquidity provision and natural micro-volatility.

- **Hawkes-Driven Aggressive Flow:** A univariate Hawkes process produces bursts of market orders that rapidly consume depth, mimicking liquidity shocks and creating volatility clustering.

The simulation uses an **event-driven framework** rather than fixed time steps. The system clock advances to the next agent arrival or Hawkes event, allowing high-frequency market interactions to be represented efficiently. The model is designed to reproduce empirical market patterns including fat-tailed returns, spread dynamics, and volatility clustering while maintaining a stable limit order book under normal conditions.

## My Contribution

I designed and implemented the **agent-based component** of the simulator, including trading strategies and decision logic for:

- **High-Frequency Traders (HFT):** inventory-aware liquidity provision and high-frequency order submission.
- **Chartists:** momentum-driven trading behavior based on recent price dynamics.
- **Fundamentalists:** value-based trading decisions driven by deviations between market prices and fundamental value.

I also analyzed the resulting agent behavior and co-authored the final project report.
