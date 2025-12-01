# [Perceptron convergence notebooks](https://chatgpt.com/c/692de812-4bc0-832b-bec4-0b20ff4b523e)

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