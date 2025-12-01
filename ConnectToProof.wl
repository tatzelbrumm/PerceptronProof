(* ::Package:: *)

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

