function K = dipolarkernel(t,r)
 K = zeros(length(t),length(r));
 dr = r(2)-r(1);
 dt = t(2)-t(1);
 for j = 1 : length(r)
   for i = 1 : length(t)
     for z = 0 : 100
       a = 52.04/((r(1)+ j*dr)^3);
       b = (1-3*z^2)*i*dt;
       K(i,j) = cos(a*b);
     end
   end
 end
end