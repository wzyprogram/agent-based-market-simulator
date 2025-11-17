from core.simulation import inject_simulation
from visual import plot_l1, plot_heatmap, plot_depth_curve
from core.structures import Parameters

params = Parameters()
data = inject_simulation(params, steps=300)

plot_l1(data["midprices"], data["best_bids"], data["best_asks"])
plot_heatmap(data["lob_history"])
# plot_depth_curve(data["final_lob"])

