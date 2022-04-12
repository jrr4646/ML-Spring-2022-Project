clear
%POD dynamics analysis

L         = 437.58;  %total length of core
n         = 8;       %Cells per assembly
Assem     = 34;      %Number of assemblies
N         = n*Assem; %Total cells for each energy group
xH        = L/N;     %Width of each mesh cell

StartX    = -L/2;
EndX      = L/2;
x         = StartX:xH:EndX;
StartTime = 0;
EndTime   = 0.00003;
tH        = 2*10^-9;
TimeSteps = int32(EndTime/tH);
vt        = StartTime:tH:EndTime;
print     = 5;                      %How often sol prints to matrix
vttrim    = StartTime:tH*print:EndTime;

trim = readmatrix('FOM_4.5.txt');
u0   = readmatrix('u0_4.5.txt');
RHS  = readmatrix('RHS_4.5.txt');
size = size(trim);

basis = 6;   %Number of orthogonal modes to take

G = zeros(size(2),basis+10);
for i=1:size(2)
   for j=1:basis+10
     G(i,j) = normrnd(0,1);
   end
end

Y     = (trim*trim.')*trim*G; %Rand subspace of FOM
[Q,R] = qr(Y);
Q     = Q(:,1:basis);
B     = Q.'*trim;

[Uy,Sy,Vy] = svd(B);  %SVD of subspace of snapshot matrix

phi            = Q*Uy;
phi(1,:)       = 0;
phi(2*(N+1),:) = 0;

a = (phi.')*u0; %Initial time coef
u = phi;

phi = phi(any(phi,2),:); %trimming zeros
phi = phi(:,any(phi,1));

Rlo = (phi.')*RHS*phi;

Low      = zeros(2*(N+1),TimeSteps+1);  %Low order solution
Lowtrim  = zeros(2*(N+1),TimeSteps/print+1);

for i=1:basis
Low(:,1) = Low(:,1) + u(:,i)*a(i);
Lowtrim(:,1) = Lowtrim(:,1) + u(:,i)*a(i);
end
Low(1,:)     = 0;
Lowtrim(1,:) = 0;

mat2 = (eye(basis)-0.5*tH*Rlo)\(eye(basis)+0.5*tH*Rlo);

for i=1:TimeSteps
    a = mat2*a;
    for j=1:basis
        Low(:,i+1) = Low(:,i+1) + u(:,j)*a(j);
    end
    k = int32(mod(i,print));
    if k==0
      Lowtrim(2:N,(i-1)/print+1)       = Low(2:N,i+1);
      Lowtrim(N+3:2*N+1,(i-1)/print+1) = Low(N+3:2*N+1,i+1);
    end 
end
if print==1
    Lowtrim = [Low(:,1) Lowtrim];
    Lowtrim(:,TimeSteps/print+2) = [];
end

diff   = abs(trim-Lowtrim);
L2diff = zeros(1,size(2));
L2trim = zeros(1,size(2));
for i=1:size(2)
    L2diff(i) = norm(diff(:,i))^2;
    L2trim(i) = norm(trim(:,i))^2;
end

error = sqrt(sum(L2diff))/sqrt(sum(L2trim));

disp("print every " + print + " timesteps");
disp("Number of basis functions: " + basis);
disp("L2 norm error from low order approx: " + error);

%High order fast flux
figure()
surf(vttrim,x,trim(1:N+1,:)); shading interp, colormap(hot);
title('Fast flux high order')
xlabel('time (s)')
ylabel('distance(x)')
zlabel('flux')
%Low order fast flux
figure()
surf(vttrim,x,Lowtrim(1:N+1,:)); shading interp, colormap(hot);
title('Fast flux low order')
xlabel('time (s)')
ylabel('distance(x)')
zlabel('flux')

%High order thermal flux
figure()
surf(vttrim,x,trim(N+2:2*(N+1),:)); shading interp, colormap(hot);
title('Thermal flux high order')
xlabel('time (s)')
ylabel('distance(x)')
zlabel('flux')
%Low order thermal flux
figure()
surf(vttrim,x,Lowtrim(N+2:2*(N+1),:)); shading interp, colormap(hot);
title('Thermal flux low order')
xlabel('time (s)')
ylabel('distance(x)')
zlabel('flux')

figure()
hold on
xlabel('x')
ylabel('phi(x)')
title('Fast flux high order')
ylim([0 1.5E14])
for i=1:10
        cla
        plot(x,trim(1:N+1,1),'r')
        plot(x,trim(1:N+1,i),'b')
        xline(-193.05)
        xline(193.05)
        pause(0.5);
end
for i=11:TimeSteps/print+1
  c = int32(mod(i,10));
  if c==1
    cla
    plot(x,trim(1:N+1,1),'r')
    plot(x,trim(1:N+1,i),'b')
    xline(-193.05)
    xline(193.05)
    pause(0.001);
  end
end
hold off

%Low Order fast flux
figure()
hold on
xlabel('x')
ylabel('phi(x)')
title('Fast flux low order')
ylim([0 1.5E14])
for i=1:10
        cla
        plot(x,Lowtrim(1:N+1,1),'r')
        plot(x,Lowtrim(1:N+1,i),'b')
        xline(-193.05)
        xline(193.05)
        pause(0.5);
end
for i=11:TimeSteps/print+1
  c = int32(mod(i,10));
  if c==1
    cla
    plot(x,Lowtrim(1:N+1,1),'r')
    plot(x,Lowtrim(1:N+1,i),'b')
    xline(-193.05)
    xline(193.05)
    pause(0.001);
  end
end
hold off

%High order Thermal flux
figure()
hold on
xlabel('x')
ylabel('phi(x)')
title('Thermal flux high order')
for i=1:TimeSteps/print+1
    c = int32(mod(i,10));
    if c==1
  cla
  plot(x,trim(N+2:2*(N+1),1),'r')
  plot(x,trim(N+2:2*(N+1),i),'b')
  xline(-193.05)
  xline(193.05)
  pause(0.001);
    end
end

hold off
%Low order Thermal flux
figure()
hold on
xlabel('x')
ylabel('phi(x)')
title('Thermal flux low order')
for i=1:TimeSteps/print+1
    c = int32(mod(i,10));
    if c==1
  cla
  plot(x,Lowtrim(N+2:2*(N+1),1),'r')
  plot(x,Lowtrim(N+2:2*(N+1),i),'b')
  xline(-193.05)
  xline(193.05)
  pause(0.001);
    end
end
hold off
