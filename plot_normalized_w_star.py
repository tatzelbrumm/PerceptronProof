import numpy as np
import matplotlib.pyplot as plt

# Convert history to arrays
w_hist = np.array(history["w"])             # shape: (T, 2)
w_dot_wstar = np.array(history["w_dot_wstar"])
w_norm_sq = np.array(history["w_norm_sq"])

# Avoid division by zero
ratio = w_dot_wstar / w_norm_sq

plt.figure()
plt.plot(ratio, marker="o")
plt.xlabel("Update step t")
plt.ylabel("(w_t · w*) / (w_t · w_t)")
plt.title("Evolution of (w_t · w*) / ||w_t||²")
plt.show()

