(* ::Package:: *)

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

