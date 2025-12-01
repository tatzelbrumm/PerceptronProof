(* ::Package:: *)

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

