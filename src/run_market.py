from core.structures import Parameters
from core.simulation import inject_simulation
import matplotlib.pyplot as plt

params = Parameters()
prices = inject_simulation(params, steps=300)

plt.plot(prices)
plt.title("Simulated Midprice Evolution")
plt.xlabel("Time Step")
plt.ylabel("Price")
plt.show()