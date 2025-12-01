(* ::Package:: *)

resultXor = perceptronTrain[Xxor, yxor, 50];

wXor = resultXor["wFinal"];
bXor = resultXor["bFinal"];
mistakesXor = resultXor["mistakesPerEpoch"];
totalUpdatesXor = resultXor["totalUpdates"];

wXor
bXor
totalUpdatesXor
mistakesXor

