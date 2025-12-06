from matplotlib.animation import FuncAnimation
from IPython.display import HTML

# Make sure we have numpy arrays
w_hist = np.array(history["w"])   # shape (T, 2)
b_hist = np.array(history["b"])   # length T

# Set up the figure and base scatter of the data
fig, ax = plt.subplots()

ax.scatter(X[y == 1, 0], X[y == 1, 1], label="+1")
ax.scatter(X[y == -1, 0], X[y == -1, 1], label="-1")

# Fix axis limits so they don't jump during animation
x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
ax.set_xlim(x_min, x_max)
ax.set_ylim(y_min, y_max)

ax.set_xlabel("x1")
ax.set_ylabel("x2")
ax.legend()

# Line object for the decision boundary
(line,) = ax.plot([], [])

# Quiver for the normal vector (w_t) from the origin
q = ax.quiver(0, 0, 0, 0, angles="xy", scale_units="xy", scale=1)

def init():
    line.set_data([], [])
    q.set_UVC(0, 0)
    ax.set_title("Perceptron updates")
    return line, q

def update(frame):
    w = w_hist[frame]
    b = b_hist[frame]

    # Compute decision boundary line
    x_vals = np.linspace(x_min, x_max, 200)

    if w[1] != 0:
        y_vals = -(w[0] * x_vals + b) / w[1]
    else:
        # Vertical line: x = -b / w[0]
        x_vals = np.full_like(x_vals, -b / w[0])
        y_vals = np.linspace(y_min, y_max, 200)

    line.set_data(x_vals, y_vals)

    # Update normal vector from the origin (direction of w_t)
    q.set_UVC(w[0], w[1])

    # Ratio (w_t · w*) / (w_t · w_t)
    wt_dot_wstar = w @ w_star
    wt_norm_sq = w @ w
    ratio = wt_dot_wstar / wt_norm_sq

    ax.set_title(f"Step {frame + 1}, ratio = {ratio:.3f}")

    return line, q

ani = FuncAnimation(
    fig,
    update,
    frames=len(w_hist),
    init_func=init,
    blit=True,
    interval=200,  # ms between frames
)

plt.close(fig)  # avoid double static figure in some Jupyter setups

HTML(ani.to_jshtml())

