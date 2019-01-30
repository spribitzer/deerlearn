function [rout, Pout] = prepPofR(rin,Pin,tmaxin,tref,nPoints)

currentscale = (tmaxin/tref)^(1/3);

rmax = max(rin)*currentscale;
rmin = min(rin)*currentscale;

rout = linspace(rmin,rmax,nPoints);

Pout = zeros(1,nPoints);

subset = and(rout >= min(rin), rout <= max(rin));

rint = rout(subset);

Pint = interp1(rin,Pin,rint); 

Pout(subset) = Pint;


