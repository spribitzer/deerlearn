clc

r1max = 3;
r2max = 6;

t1max = 4;
t2max = 4*(r2max/r1max)^3;

disp(r2max/r1max);

rscale = (t2max/t1max)^(1/3);

disp(rscale)

disp(r1max*rscale)




