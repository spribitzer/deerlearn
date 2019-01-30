function [TraceOut] = prepDEERSignal(TraceIn, nPoints)

nIn = length(TraceIn);
xIn = 1:nIn;

xOut = linspace(1,nIn,nPoints);

TraceOut = interp1(xIn,TraceIn,xOut);  
