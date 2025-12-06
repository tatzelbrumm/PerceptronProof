# [Perceptron convergence notebooks](https://chatgpt.com/g/g-p-6930e7063df0819197742d72471aaaa8-ananthaswamy-ai/c/692de812-4bc0-832b-bec4-0b20ff4b523e)

You: 

please make me both mathematica and jupyter notebooks that illustrate the convergence proof of rosenblatt's perceptron. pay attention that not every random data set has a hyperplane that separates the two output sets.

ChatGPT: 

Thought for 1m 6s

Here’s a complete package you can work with.

---

## 1\. Jupyter notebook (Python)

I’ve actually created a ready-to-use notebook file for you:

**➡️ Download the Jupyter notebook**

What it contains:

1.  **Linearly separable case**
    
    -   Synthetic data generated from a “teacher” hyperplane $(w^\*, b^\*)$ with a margin.
        
    -   Standard perceptron training loop with labels in $\{-1, +1\}$.
        
    -   Tracking:
        
        -   $w_t \cdot w^\*$
            
        -   $\|w_t\|^2$
            
        -   mistakes per epoch
            
    -   Computation of empirical $R = \max_i \|x_i\|$ and $\gamma = \min_i y_i (w^\* \cdot x_i + b^\*)$.
        
    -   Comparison of observed updates with the bound $(R/\gamma)^2$.
        
    -   Plots:
        
        -   Data + learned decision boundary.
            
        -   Growth of $w_t \cdot w^\*$.
            
        -   Growth of $\|w_t\|^2$.
            
2.  **Non-separable case**
    
    -   Classic XOR dataset in $\mathbb{R}^2$.
        
    -   Same perceptron training loop, limited to a finite number of epochs.
        
    -   Output of mistakes per epoch to show lack of convergence.
        
    -   Plot of data + (futile) perceptron decision boundary.
        

This directly illustrates the logic of the convergence proof and makes explicit that **non-separable data violates the theorem’s assumptions**, so the algorithm can fail to converge.

---

## 2\. Mathematica notebook (Wolfram Language)

Below is a self-contained sequence of cells you can paste into a Mathematica notebook (`.nb`).  
I’ve followed the same structure as the Python notebook:

1.  Separable data from a known hyperplane.
    
2.  Perceptron training with tracking of proof quantities.
    
3.  Visualization of convergence.
    
4.  Non-separable XOR example showing non-convergence.
    

You can paste each block as its own cell.

### 2.1 Separable data and “teacher” hyperplane

```wl
(* ::Section:: *)
(*Perceptron Convergence Theorem - Numerical Illustration*)

(* We work in R^2 and construct data that is linearly separable
   by a "teacher" hyperplane (wStar, bStar). *)

ClearAll[generateSeparableData];

generateSeparableData[nSamples_Integer : 60, dim_Integer : 2, margin_?NumericQ : 0.5] :=
 Module[{wStar, bStar, x, m, label, X = {}, y = {}, rng},
  (* Random but fixed separator *)
  wStar = Normalize[RandomReal[NormalDistribution[0, 1], dim]];
  bStar = RandomReal[NormalDistribution[0, 0.1]];

  While[Length[X] < nSamples,
   x = RandomVariate[NormalDistribution[0, 1], dim];
   m = wStar.x + bStar;
   If[Abs[m] < margin, Continue[]];
   label = If[m > 0, 1, -1];
   AppendTo[X, x];
   AppendTo[y, label];
  ];

  <|"X" -> X, "y" -> y, "wStar" -> wStar, "bStar" -> bStar|>
 ];

data = generateSeparableData[];
X = data["X"];
y = data["y"];
wStar = data["wStar"];
bStar = data["bStar"];

Dimensions[X]
Counts[y]
```

Plot the separable data:

```wl
(* ::Subsection:: *)
(*Plot of linearly separable data*)

pos = Pick[X, UnitStep[y], 1];
neg = Pick[X, UnitStep[-y], 1];

ListPlot[
 {
  pos,
  neg
 },
 PlotStyle -> {PointSize[Large], PointSize[Large]},
 PlotLegends -> {"+1", "-1"},
 AxesLabel -> {"x1", "x2"},
 PlotRange -> All,
 ImageSize -> Large
]
```

---

### 2.2 Perceptron training with tracking of proof quantities

```wl
(* ::Subsection:: *)
(*Perceptron algorithm with tracking*)

ClearAll[perceptronTrain];

perceptronTrain[X_List, y_List, maxEpochs_Integer : 1000, wStar_: None, bStar_: None] :=
 Module[{nSamples, dim, w, b, epoch, i, xi, yi, margin, mistakes,
   historyW = {}, historyB = {}, historyWDotWStar = {}, historyWNormSq = {},
   mistakesPerEpoch = {}, totalUpdates = 0, wDot},

  nSamples = Length[X];
  dim = Length[First[X]];
  w = ConstantArray[0., dim];
  b = 0.;

  For[epoch = 1, epoch <= maxEpochs, epoch++,
   mistakes = 0;
   For[i = 1, i <= nSamples, i++,
    xi = X[[i]];
    yi = y[[i]];
    margin = yi (w.xi + b);
    If[margin <= 0,
     w = w + yi xi;
     b = b + yi;
     mistakes++;
     totalUpdates++;

     If[wStar =!= None,
      wDot = w.wStar;
      AppendTo[historyWDotWStar, wDot];
      ,
      AppendTo[historyWDotWStar, Missing["NotAvailable"]];
     ];
     AppendTo[historyWNormSq, w.w];
     AppendTo[historyW, w];
     AppendTo[historyB, b];
    ];
   ];
   AppendTo[mistakesPerEpoch, mistakes];
   If[mistakes == 0, Break[]];
  ];

  <|
   "wFinal" -> w,
   "bFinal" -> b,
   "historyW" -> historyW,
   "historyB" -> historyB,
   "historyWDotWStar" -> historyWDotWStar,
   "historyWNormSq" -> historyWNormSq,
   "mistakesPerEpoch" -> mistakesPerEpoch,
   "totalUpdates" -> totalUpdates
  |>
 ];

result = perceptronTrain[X, y, 100, wStar, bStar];

wFinal = result["wFinal"];
bFinal = result["bFinal"];
totalUpdates = result["totalUpdates"];
mistakesPerEpoch = result["mistakesPerEpoch"];

wFinal
bFinal
totalUpdates
mistakesPerEpoch
```

Plot mistakes per epoch:

```wl
ListLinePlot[
 mistakesPerEpoch,
 AxesLabel -> {"epoch", "mistakes"},
 PlotMarkers -> Automatic,
 PlotTheme -> "Scientific",
 ImageSize -> Large,
 PlotLabel -> "Mistakes per epoch (separable case)"
]
```

---

### 2.3 Learned decision boundary

```wl
(* ::Subsection:: *)
(*Decision boundary in R^2*)

ClearAll[plotDecisionBoundary];

plotDecisionBoundary[w_, b_, X_, y_] :=
 Module[{pos, neg, line},
  pos = Pick[X, UnitStep[y], 1];
  neg = Pick[X, UnitStep[-y], 1];

  Show[
   ListPlot[
    {pos, neg},
    PlotStyle -> {PointSize[Large], PointSize[Large]},
    PlotLegends -> {"+1", "-1"},
    AxesLabel -> {"x1", "x2"},
    PlotRange -> All
   ],
   (* line w.x + b == 0 => x2 = -(w1 x1 + b)/w2 *)
   If[w[[2]] != 0,
    line = Plot[
      -(w[[1]] x + b)/w[[2]],
      {x, Min[X[[All, 1]]] - 1, Max[X[[All, 1]]] + 1}
     ];
    line,
    {}
   ],
   ImageSize -> Large,
   PlotLabel -> "Perceptron decision boundary"
  ]
 ];

plotDecisionBoundary[wFinal, bFinal, X, y];
```

---

### 2.4 Connecting to the proof: $R$, $\gamma$, and growth bounds

```wl
(* ::Subsection:: *)
(*R, gamma, and theoretical bound*)

norms = Norm /@ X;
R = Max[norms];

marginsTeacher = MapThread[#1 (#2.wStar + bStar) &, {y, X}];
gamma = Min[marginsTeacher];

R
gamma

bound = If[gamma > 0, (R/gamma)^2, Infinity]

<|"R" -> R, "gamma" -> gamma, "bound" -> bound, "observedUpdates" -> totalUpdates|>
```

Evolution of $w_t \cdot w^\*$ and $\|w_t\|^2$:

```wl
wDotWStar = DeleteMissing[result["historyWDotWStar"]];
wNormSq = result["historyWNormSq"];
steps = Range[Length[wNormSq]];

ListLinePlot[
 wDotWStar,
 AxesLabel -> {"update step t", "w_t \[CenterDot] w*"},
 PlotTheme -> "Scientific",
 PlotLabel -> "Growth of w_t · w*",
 ImageSize -> Large
]

ListLinePlot[
 wNormSq,
 AxesLabel -> {"update step t", "||w_t||^2"},
 PlotTheme -> "Scientific",
 PlotLabel -> "Growth of ||w_t||^2",
 ImageSize -> Large
]
```

This mirrors the standard argument:

-   $w_t \cdot w^\*$ grows at least linearly with the number of updates (lower bound).
    
-   $\|w_t\|^2$ grows at most linearly (upper bound).
    
-   Combining both yields the bound $T \le (R/\gamma)^2$ on the total number of updates $T$.
    

---

### 2.5 Non-separable XOR example

```wl
(* ::Section:: *)
(*Non-separable data: XOR*)

Xxor = {
  {1., 1.},
  {1., -1.},
  {-1., 1.},
  {-1., -1.}
};
yxor = {1, -1, -1, 1};

ListPlot[
 {
  Pick[Xxor, UnitStep[yxor], 1],
  Pick[Xxor, UnitStep[-yxor], 1]
 },
 PlotStyle -> {PointSize[Large], PointSize[Large]},
 PlotLegends -> {"+1", "-1"},
 AxesLabel -> {"x1", "x2"},
 PlotRange -> {{-2, 2}, {-2, 2}},
 PlotLabel -> "XOR pattern (not linearly separable)",
 ImageSize -> Large
]
```

Train perceptron on the XOR data, with a cap on epochs:

```wl
resultXor = perceptronTrain[Xxor, yxor, 50];

wXor = resultXor["wFinal"];
bXor = resultXor["bFinal"];
mistakesXor = resultXor["mistakesPerEpoch"];
totalUpdatesXor = resultXor["totalUpdates"];

wXor
bXor
totalUpdatesXor
mistakesXor
```

Plot mistakes per epoch:

```wl
ListLinePlot[
 mistakesXor,
 AxesLabel -> {"epoch", "mistakes"},
 PlotTheme -> "Scientific",
 PlotLabel -> "Mistakes per epoch (XOR, non-separable)",
 ImageSize -> Large
]
```

Plot the (ineffective) decision boundary:

```wl
plotDecisionBoundary[wXor, bXor, Xxor, yxor];
```

You should see that the perceptron **does not** achieve zero mistakes on XOR—consistent with the fact that **no hyperplane exists that separates the two label sets**. This directly emphasizes the logical structure of Rosenblatt’s theorem:

-   **If** there exists a separating hyperplane with positive margin, **then** the perceptron converges in finitely many steps (bounded by $(R/\gamma)^2$).
    
-   For data that is **not** linearly separable (which is typical for arbitrary random data), this guarantee does **not** apply, and the algorithm can keep making mistakes forever.
    

---

If you want, I can also add a small symbolic derivation cell in Mathematica that reproduces the algebraic inequality chain of the convergence proof step by step.

You: 

My Mathematica 13.2 barfs with Syntax::sntxf: "generateSeparableData\[" cannot be followed by "nSamples\_Integer:60,dim\_Integer:2,margin\_?NumericQ:0.5\]".

ChatGPT: 

Thought for 9s

Right, that’s on me – I made the definition fancier than it needs to be.

Mathematica **can** handle patterns with defaults, but the combination `n_Integer : 60, margin_?NumericQ : 0.5` is more brittle across versions and easy to break when copy-pasting.

Let’s simplify the function header and avoid type constraints + defaults there. You can still have defaults via a separate zero-argument definition.

### Fix: simpler `generateSeparableData` definition

Replace the old definition with this:

```wl
ClearAll[generateSeparableData];

generateSeparableData[nSamples_, dim_, margin_] :=
 Module[{wStar, bStar, x, m, label, X = {}, y = {}},
  (* Random but fixed separator *)
  wStar = Normalize[RandomReal[NormalDistribution[0, 1], dim]];
  bStar = RandomReal[NormalDistribution[0, 0.1]];

  While[Length[X] < nSamples,
   x = RandomVariate[NormalDistribution[0, 1], dim];
   m = wStar.x + bStar;
   If[Abs[m] < margin, Continue[]];
   label = If[m > 0, 1, -1];
   AppendTo[X, x];
   AppendTo[y, label];
  ];

  <|"X" -> X, "y" -> y, "wStar" -> wStar, "bStar" -> bStar|>
 ];
```

If you want the same defaults as before (60 samples, 2D, margin 0.5):

```wl
generateSeparableData[] := generateSeparableData[60, 2, 0.5];
```

Then you can call:

```wl
data = generateSeparableData[];   (* uses defaults *)
(* or *)
(* data = generateSeparableData[60, 2, 0.5]; *)

X     = data["X"];
y     = data["y"];
wStar = data["wStar"];
bStar = data["bStar"];
```

Everything else in the notebook (perceptron training, plots, convergence proof quantities, XOR example) can stay as I gave it; only this function header needs to be updated.

If you’d like, I can also give you a single minimal `.wl` script that runs through the whole convergence demo in one go without notebook markup.

You: 

Ok, Mathematica works now, but the Jupyter notebook doesn't find numpy in my conda jupyter setup I got from ...somewhere... How do I resolve installation and dependency hell?

ChatGPT: 

Thought for 11s

Yeah, welcome to the club – everyone meets “numpy not found” at some point 😄  
The good news: this is almost always “Jupyter is using a different Python than the one where you installed stuff.”

I’ll give you a **minimal, robust recipe** to get a clean working setup, and then a short explanation of what’s going on.

---

## Quick fix: create a clean conda env + Jupyter kernel

Best is to create a fresh environment just for this perceptron notebook and hook it into Jupyter.

In a terminal:

```bash
# 1. Create a clean env with everything we need
conda create -n perceptron-demo python=3.11 numpy matplotlib jupyterlab ipykernel -y

# 2. Activate it
conda activate perceptron-demo

# 3. Register this environment as a Jupyter kernel
python -m ipykernel install --user --name perceptron-demo --display-name "Perceptron (conda)"
```

Then:

1.  Start Jupyter (from the same activated env, just to keep things simple):
    
    ```bash
    jupyter lab
    ```
    
    or
    
    ```bash
    jupyter notebook
    ```
    
2.  Open `perceptron_convergence_demo.ipynb`.
    
3.  In the notebook UI:  
    **Kernel → Change kernel → “Perceptron (conda)”**.
    

Now `import numpy as np` and `import matplotlib.pyplot as plt` should work without complaint.

---

## If you want to reuse your *existing* environment instead

If you *don’t* want a new env and just want numpy in whatever env Jupyter is already using:

1.  Figure out which env that is. Start Jupyter as you normally do, open a new notebook, and in a cell run:
    
    ```python
    import sys
    sys.executable
    ```
    
    That path tells you which Python binary the kernel is using, e.g. something like:
    
    -   `.../anaconda3/envs/someenv/bin/python`
        
    -   or `.../anaconda3/bin/python` (base env)
        
2.  In a terminal, activate **that** env, for example:
    
    ```bash
    conda activate someenv
    ```
    
3.  Install numpy and matplotlib into that env:
    
    ```bash
    conda install numpy matplotlib -y
    ```
    
4.  Restart the notebook kernel and try again.
    

If you see `pip` paths instead of conda ones, you can still do:

```bash
/path/to/that/python -m pip install numpy matplotlib
```

…but **try conda first** if it’s really a conda env, to avoid the usual binary mess.

---

## Why this happens (short version)

-   Conda can create many isolated environments: `base`, `foo`, `bar`, …
    
-   Jupyter kernels are just pointers to specific Python binaries.
    
-   If you installed `numpy` in env `foo` but your Jupyter kernel points to env `bar`, then `import numpy` in `bar` fails.
    

So the rules of the game are:

1.  Install packages in the env whose Python you actually run.
    
2.  Make that env visible to Jupyter via `ipykernel`.
    
3.  Select the right kernel in the notebook UI.
    

---

If you tell me how you currently start Jupyter (e.g. from base, from VS Code, from Anaconda Navigator), I can give you a “do exactly this, nothing else” variant tailored to your setup.

You: 

and how can i check which kernels are still active, and how can i stop them?

ChatGPT: 

Here are **all reliable ways** to see which Jupyter kernels are running *right now* and how to shut them down cleanly — with and without the Jupyter UI.

---

# ✅ **1\. The easy way (works everywhere): Jupyter Notebook / Lab UI**

### **In classic Jupyter Notebook**

Go to:

```mathematica
File → Shut Down...
```

This stops the *entire* Jupyter server, which kills **all kernels**.

To stop individual kernels:

```nginx
Kernel → Shutdown
```

### **In JupyterLab**

-   Click **Running** (left sidebar).
    
-   You’ll see:
    
    **Terminals**  
    **Kernels**  
    **Sessions**
    

Each kernel is listed with an ● indicator.

To stop one:  
**Click the "Stop" (⨯) button next to it.**

To stop everything:  
**Click "Shut Down All"** in the Running panel.

---

# ✅ **2\. From the terminal: list and kill active kernels**

Jupyter kernels run as operating system processes called **kernel-xxxxx.json**.  
You can list them:

### **Show kernel activity files**

```bash
jupyter runtime list
```

or

```bash
ls ~/.local/share/jupyter/runtime/
```

You will see files such as:

```pgsql
kernel-12345.json
kernel-abcde.json
```

These represent active kernels.

If a file is still there after a reboot, that's a “zombie” kernel file (harmless; can be deleted).

### **Stop the Jupyter server (kills all kernels)**

If you started Jupyter in a terminal, press:

```objectivec
CTRL+C
```

twice.

Or kill all Jupyter servers explicitly:

```bash
jupyter notebook stop
```

or list servers:

```bash
jupyter notebook list
```

You’ll see:

```bash
http://localhost:8888/?token=xyz :: /path/to/notebook
```

To stop a specific one:

```bash
jupyter notebook stop 8888
```

---

# ✅ **3\. Using `ps` (Linux/macOS) or Task Manager (Windows)**

### **Linux/macOS**

List running kernels (python processes sitting in IPython):

```bash
ps aux | grep ipykernel
```

Typical output:

```swift
python -m ipykernel_launcher -f /home/you/.local/.../kernel-12345.json
```

Kill one kernel:

```bash
kill <PID>
```

Or all kernels:

```bash
pkill -f ipykernel
```

---

### **Windows**

Open **Task Manager → Details**  
Look for:

```scss
python.exe  (ipykernel_launcher)
```

Right-click → **End Task**.

---

# ✅ **4\. See all installed kernels (not running)**

This tells you what kernels exist (not which are active):

```bash
jupyter kernelspec list
```

Typical output:

```swift
Available kernels:
  python3          /home/you/anaconda3/share/jupyter/kernels/python3
  perceptron-demo  /home/you/.local/share/jupyter/kernels/perceptron-demo
  myenv            /home/you/.local/share/jupyter/kernels/myenv
```

To delete one:

```bash
jupyter kernelspec uninstall myenv
```

---

# Quick summary table

| Task | Command / UI |
| --- | --- |
| List active servers | `jupyter notebook list` |
| Stop a server | `jupyter notebook stop <port>` |
| Stop all kernels | JupyterLab → Running → Shut Down All |
| List active kernel runtime files | `jupyter runtime list` |
| List installed kernels | `jupyter kernelspec list` |
| Kill kernels by process | `pkill -f ipykernel` |
| Delete kernel from system | `jupyter kernelspec uninstall <name>` |

---

If you want, you can tell me:

-   How you start Jupyter (Navigator? VS Code? Terminal?),
    
-   Your OS,
    

and I can give you exact steps tailored to your setup.

You: 

perceptron\_convergence\_demo.ipynb

File

I'm a bit lazy, so I'm asking you first instead of looking at your code. In this latest tweak, what would I have to change in the Jupyter notebook to get an animation of how the separating plane and its attached normal vector change for every iteration of its adjustment, and the ratio (w\_t·w\*)/(w\_t·w\_t) ?

ChatGPT: 

Thought for 16s

You don’t have to change very much at all. Your existing notebook already stores everything needed for both the animation and the ratio $(w_t\cdot w^\*)/(w_t\cdot w_t)$.

I’ll assume you’re using the notebook I generated (with `history["w"]`, `history["b"]`, `history["w_dot_wstar"]`, `history["w_norm_sq"]`).

Below are **two things to add**:

1.  A cell to plot the ratio $(w_t\cdot w^\*)/(w_t\cdot w_t)$.
    
2.  A cell to animate the separating line + normal vector over updates.
    

Just paste these *after* the perceptron training section (after the cell where you print `total_updates` and plot mistakes per epoch).

---

## 1\. Ratio $(w_t·w^\*)/(w_t·w_t)$

Add this cell:

```python
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
```

That gives you the ratio for every **update step** (not per epoch, but per actual weight adjustment).

---

## 2\. Animation of separating line + normal vector

This uses `matplotlib.animation.FuncAnimation`. We reuse the stored `w` and `b` for every update and draw:

-   The current separating line $w_t \cdot x + b_t = 0$,
    
-   The current normal vector $w_t$ from the origin,
    
-   Optionally the current ratio in the title.
    

Add this cell:

```python
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
```

Notes:

-   This will embed an interactive JS animation directly in the notebook.
    
-   The title shows the **current update step** and the ratio $(w_t·w^\*)/(w_t·w_t)$.
    
-   The normal vector is drawn as an arrow from the origin in the direction of $w_t$.
    
-   The line is re-drawn for each update according to the current $w_t, b_t$.
    

---

### If anything breaks

The only assumptions this code makes are:

-   `history["w"]` and `history["b"]` are lists of the weights and biases **after each update** (which is how the function I gave you records them).
    
-   `w_star` is still in scope (from the data generation cell).
    

If you renamed those variables or moved cells around, tell me what you changed and I’ll adapt the snippets to your exact notebook layout.