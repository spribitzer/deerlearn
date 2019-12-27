clear

r = linspace(0,7,256);
r0 = 4; % nm
fwhm = 0.4;
P = gaussian(r,r0,fwhm);
dr = r(2)-r(1);
nr = numel(r);
nt = nr;

t = linspace(0,5,nt);

K = dipolarkernel(t,r);
S = K*P';

scale = 1.2;
r2 = r*scale;
dr2 = r2(2)-r2(1);

K2 = dipolarkernel(t,r2);
S2 = K2*P'*dr/dr2;

subplot(2,1,2);
plot(r,P);
axis tight

subplot(2,1,1);
plot(t,S,t/scale^3,S2);
axis tight