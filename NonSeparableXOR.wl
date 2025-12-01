(* ::Package:: *)

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

