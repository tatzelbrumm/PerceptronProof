(* ::Package:: *)

(* ::Subsection:: *)
(*Perceptron Convergence Theorem-Numerical Illustration*)


(* ::Text:: *)
(*We work in \[DoubleStruckCapitalR]\.b2 and construct data that is linearly separable by a "teacher" hyperplane (SuperStar[w],SuperStar[b]).*)


ClearAll[generateSeparableData];

generateSeparableData[nSamples_Integer:60,dim_Integer:2,margin_?NumericQ:0.5]:=Module[{wStar,bStar,x,m,label,X={},y={},rng},(*Random but fixed separator*)wStar=Normalize[RandomReal[NormalDistribution[0,1],dim]];
bStar=RandomReal[NormalDistribution[0,0.1]];
While[Length[X]<nSamples,x=RandomVariate[NormalDistribution[0,1],dim];
m=wStar . x+bStar;
If[Abs[m]<margin,Continue[]];
label=If[m>0,1,-1];
AppendTo[X,x];
AppendTo[y,label];];
<|"X"->X,"y"->y,"wStar"->wStar,"bStar"->bStar|>];

data=generateSeparableData[];
X=data["X"];
y=data["y"];
wStar=data["wStar"];
bStar=data["bStar"];

Dimensions[X]
Counts[y]



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
    margin = yi (w . xi + b);
    If[margin <= 0,
     w = w + yi xi;
     b = b + yi;
     mistakes++;
     totalUpdates++;

     If[wStar =!= None,
      wDot = w . wStar;
      AppendTo[historyWDotWStar, wDot];
      ,
      AppendTo[historyWDotWStar, Missing["NotAvailable"]];
     ];
     AppendTo[historyWNormSq, w . w];
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



ListLinePlot[
 mistakesPerEpoch,
 AxesLabel -> {"epoch", "mistakes"},
 PlotMarkers -> Automatic,
 PlotTheme -> "Scientific",
 ImageSize -> Large,
 PlotLabel -> "Mistakes per epoch (separable case)"
]



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



(* ::Subsection:: *)
(*R, gamma, and theoretical bound*)


norms = Norm /@ X;
R = Max[norms];

marginsTeacher = MapThread[#1 (#2 . wStar + bStar) &, {y, X}];
gamma = Min[marginsTeacher];

R
gamma

bound = If[gamma > 0, (R/gamma)^2, Infinity]

<|"R" -> R, "gamma" -> gamma, "bound" -> bound, "observedUpdates" -> totalUpdates|>



wDotWStar = DeleteMissing[result["historyWDotWStar"]];
wNormSq = result["historyWNormSq"];
steps = Range[Length[wNormSq]];

ListLinePlot[
 wDotWStar,
 AxesLabel -> {"update step t", "w_t \[CenterDot] w*"},
 PlotTheme -> "Scientific",
 PlotLabel -> "Growth of w_t \[CenterDot] w*",
 ImageSize -> Large
]

ListLinePlot[
 wNormSq,
 AxesLabel -> {"update step t", "||w_t||^2"},
 PlotTheme -> "Scientific",
 PlotLabel -> "Growth of ||w_t||^2",
 ImageSize -> Large
]



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



resultXor = perceptronTrain[Xxor, yxor, 50];

wXor = resultXor["wFinal"];
bXor = resultXor["bFinal"];
mistakesXor = resultXor["mistakesPerEpoch"];
totalUpdatesXor = resultXor["totalUpdates"];

wXor
bXor
totalUpdatesXor
mistakesXor



ListLinePlot[
 mistakesXor,
 AxesLabel -> {"epoch", "mistakes"},
 PlotTheme -> "Scientific",
 PlotLabel -> "Mistakes per epoch (XOR, non-separable)",
 ImageSize -> Large
]



plotDecisionBoundary[wXor, bXor, Xxor, yxor];

