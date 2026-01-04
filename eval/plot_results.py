import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import curve_fit

model_sizes = np.array([0.5, 1.5, 3, 7])
fimo_perf = np.array([14.29, 20.63, 22.22, 30.16])
putnam_perf = np.array([20.83, 31.25, 37.5, 43.75])

def log_func(x, a, b):
    return a * np.log10(x) + b

# Fit logarithmic regressions
popt_fimo, _ = curve_fit(log_func, model_sizes, fimo_perf)
popt_putnam, _ = curve_fit(log_func, model_sizes, putnam_perf)

x_fit = np.logspace(-1, 2.5, 100)
y_fimo_fit = log_func(x_fit, *popt_fimo)
y_putnam_fit = log_func(x_fit, *popt_putnam)

plt.figure(figsize=(8,6))
plt.scatter(model_sizes, fimo_perf, color='gold', zorder=5, label='FIMO')
plt.plot(x_fit, y_fimo_fit, ':', color='gold')

plt.scatter(model_sizes, putnam_perf, color='red', zorder=5, label='PutnamBench')
plt.plot(x_fit, y_putnam_fit, ':', color='red')

plt.xscale('log')
plt.xticks([1, 10, 100], ['1B', '10B', '100B'])
plt.xlabel("Model Size")
plt.ylabel("Performance (%)")
plt.title("LLM Performance vs Model Size")
plt.grid(True, linestyle='--', linewidth=0.5)
plt.legend()
plt.show()
