clear
%Two group 1D diffusion

L         = 437.58;  %total length of core
n         = 8;       %Cells per assembly
Assem     = 34;      %Number of assemblies
N         = n*Assem; %Total cells for each energy group
xH        = L/N;     %Width of each mesh cell

f = @(x) 5*cos(pi*x/L)-2*cos(3*pi*x/L);

StartX    = -L/2;
EndX      = L/2;
x         = StartX:xH:EndX;
StartTime = 0;
EndTime   = 0.00003;
tH        = 2*10^-9;
TimeSteps = int32(EndTime/tH);
vt        = StartTime:tH:EndTime;
print     = 5;                      %How often sol prints to snapshot matrix
vttrim    = StartTime:tH*print:EndTime;
snapshots = 6;

for enrich=1:snapshots

if enrich==1  %3.0% enrichment
xs = zeros(10,N); 
for i=1:N
    if i>2*n && i<=(Assem-2)*n          %Fuel cell
      xs(1,i)  = 0.552152;      %tot_1
      xs(2,i)  = 1.39967;       %tot_2
      xs(3,i)  = 0.221675;      %trans1
      xs(4,i)  = 0.925992;      %trans2
      xs(5,i)  = 0.00646568;    %nuf1
      xs(6,i)  = 0.127983;      %nuf2
      xs(7,i)  = 0.524334;      %s11
      xs(8,i)  = 0.0180330;     %s12
      xs(9,i)  = 0.00157997;    %s21
      xs(10,i) = 1.30113;       %s22
    else                                %water cell
      xs(1,i)  = 0.67499;    %tot_1w
      xs(2,i)  = 1.86142;    %tot_2w
      xs(3,i)  = 0.173675;   %trans1w
      xs(4,i)  = 1.11238;    %trans2w
      xs(7,i)  = 0.646558;   %s11w
      xs(8,i)  = 0.0295918;  %s12w
      xs(9,i)  = 0.00219350; %s21w
      xs(10,i) = 1.84778;    %s22w
    end
end
    
elseif enrich==2 %3.5% enrichment
xs = zeros(10,N); 
for i=1:N
    if i>2*n && i<=(Assem-2)*n          %Fuel cell
      xs(1,i)  = 0.551699;      %tot_1
      xs(2,i)  = 1.40563;       %tot_2
      xs(3,i)  = 0.221414;      %trans1
      xs(4,i)  = 0.928014;      %trans2
      xs(5,i)  = 0.00710788;    %nuf1
      xs(6,i)  = 0.144879;      %nuf2
      xs(7,i)  = 0.523799;      %s11
      xs(8,i)  = 0.0177880;     %s12
      xs(9,i)  = 0.00169572;    %s21
      xs(10,i) = 1.29899;       %s22
    else                                %water cell
      xs(1,i)  = 0.67499;    %tot_1w
      xs(2,i)  = 1.86142;    %tot_2w
      xs(3,i)  = 0.173675;   %trans1w
      xs(4,i)  = 1.11238;    %trans2w
      xs(7,i)  = 0.646558;   %s11w
      xs(8,i)  = 0.0295918;  %s12w
      xs(9,i)  = 0.00219350; %s21w
      xs(10,i) = 1.84778;    %s22w
    end
end
    
elseif enrich==3  %4.0% enrichment
xs = zeros(10,N); 
for i=1:N
    if i>2*n && i<=(Assem-2)*n          %Fuel cell
      xs(1,i)  = 0.551257;      %tot_1
      xs(2,i)  = 1.41165;       %tot_2
      xs(3,i)  = 0.221161;      %trans1
      xs(4,i)  = 0.930358;      %trans2
      xs(5,i)  = 0.00773376;    %nuf1
      xs(6,i)  = 0.160933;      %nuf2
      xs(7,i)  = 0.523277;      %s11
      xs(8,i)  = 0.0175519;     %s12
      xs(9,i)  = 0.00180394;    %s21
      xs(10,i) = 1.29729;       %s22
    else                                %water cell
      xs(1,i)  = 0.67499;    %tot_1w
      xs(2,i)  = 1.86142;    %tot_2w
      xs(3,i)  = 0.173675;   %trans1w
      xs(4,i)  = 1.11238;    %trans2w
      xs(7,i)  = 0.646558;   %s11w
      xs(8,i)  = 0.0295918;  %s12w
      xs(9,i)  = 0.00219350; %s21w
      xs(10,i) = 1.84778;    %s22w
    end
end
    
elseif enrich==4   %4.5% enrichment
xs = zeros(10,N); 
for i=1:N
    if i>2*n && i<=(Assem-2)*n          %Fuel cell
      xs(1,i)  = 0.550819;      %tot_1
      xs(2,i)  = 1.41768;       %tot_2
      xs(3,i)  = 0.220912;      %trans1
      xs(4,i)  = 0.932959;      %trans2
      xs(5,i)  = 0.00834417;    %nuf1
      xs(6,i)  = 0.176236;      %nuf2
      xs(7,i)  = 0.52276;       %s11
      xs(8,i)  = 0.0173228;     %s12
      xs(9,i)  = 0.00190548;    %s21
      xs(10,i) = 1.29596;       %s22
    else                                %water cell
      xs(1,i)  = 0.67499;    %tot_1w
      xs(2,i)  = 1.86142;    %tot_2w
      xs(3,i)  = 0.173675;   %trans1w
      xs(4,i)  = 1.11238;    %trans2w
      xs(7,i)  = 0.646558;   %s11w
      xs(8,i)  = 0.0295918;  %s12w
      xs(9,i)  = 0.00219350; %s21w
      xs(10,i) = 1.84778;    %s22w
    end
end
    
elseif enrich==5   %5.0% enrichment  
xs = zeros(10,N); 
for i=1:N
    if i>2*n && i<=(Assem-2)*n          %Fuel cell
      xs(1,i)  = 0.550382;      %tot_1
      xs(2,i)  = 1.42373;       %tot_2
      xs(3,i)  = 0.220675;      %trans1
      xs(4,i)  = 0.935783;      %trans2
      xs(5,i)  = 0.00893773;    %nuf1
      xs(6,i)  = 0.190862;      %nuf2
      xs(7,i)  = 0.522247;      %s11
      xs(8,i)  = 0.0171097;     %s12
      xs(9,i)  = 0.00200092;    %s21
      xs(10,i) = 1.29496;       %s22
    else                                %water cell
      xs(1,i)  = 0.67499;    %tot_1w
      xs(2,i)  = 1.86142;    %tot_2w
      xs(3,i)  = 0.173675;   %trans1w
      xs(4,i)  = 1.11238;    %trans2w
      xs(7,i)  = 0.646558;   %s11w
      xs(8,i)  = 0.0295918;  %s12w
      xs(9,i)  = 0.00219350; %s21w
      xs(10,i) = 1.84778;    %s22w
    end
end

elseif enrich==6   %3.7% enrichment  
xs = zeros(10,N); 
for i=1:N
    if i>2*n && i<=(Assem-2)*n          %Fuel cell
      xs(1,i)  = 0.551521;      %tot_1
      xs(2,i)  = 1.40803;       %tot_2
      xs(3,i)  = 0.221312;      %trans1
      xs(4,i)  = 0.928916;      %trans2
      xs(5,i)  = 0.00736013;    %nuf1
      xs(6,i)  = 0.151396;      %nuf2
      xs(7,i)  = 0.523589;      %s11
      xs(8,i)  = 0.0176925;     %s12
      xs(9,i)  = 0.00173988;    %s21
      xs(10,i) = 1.29826;       %s22
    else                                %water cell
      xs(1,i)  = 0.67499;    %tot_1w
      xs(2,i)  = 1.86142;    %tot_2w
      xs(3,i)  = 0.173675;   %trans1w
      xs(4,i)  = 1.11238;    %trans2w
      xs(7,i)  = 0.646558;   %s11w
      xs(8,i)  = 0.0295918;  %s12w
      xs(9,i)  = 0.00219350; %s21w
      xs(10,i) = 1.84778;    %s22w
    end
end
end

DiffF    = zeros(1,N); %Fast Diffusion coef
FF       = zeros(1,N); %Fast nuSigF coef
RF       = zeros(1,N); %Fast sigRemov (sigT-self_scatter)
DiffT    = zeros(1,N); %Thermal Diffusion coef
FT       = zeros(1,N); %Thermal nuSigF coef
RT       = zeros(1,N); %Thermal sigRemov (sigT-self_scatter)
DownScat = zeros(1,N); %Downscatter 12
UpScat   = zeros(1,N); %Upscatter 21

for i=1:N
  DiffF(i) = 1/(3*xs(3,i));
  FF(i)    = xs(5,i);
  RF(i)    = xs(1,i)-xs(7,i);
 
  DiffT(i) = 1/(3*xs(4,i));
  FT(i)    = xs(6,i);
  RT(i)    = xs(2,i)-xs(10,i);
 
  DownScat(i) = xs(8,i);
  UpScat(i)   = xs(9,i);
end

DF     = zeros(N+1, N+1);  %stiffness matrix for (D*gradphi_j,gradphi_k)
DT     = zeros(N+1,N+1);
M      = zeros(N+1, N+1);  %mass matrix (phi_j,phi_k)
FissF  = zeros(N+1, N+1);  %fission matrix (nusigF*phi_j,phi_k)
FissT  = zeros(N+1, N+1);
RemovF = zeros(N+1, N+1);  %Removal matrix (sigR*phi_j,phi_k)
RemovT = zeros(N+1, N+1);
UpS    = zeros(N+1, N+1);  %Upscatter
DownS  = zeros(N+1, N+1);  %Downscatter

for i=2:N
    DF(i,i)     = xH*DiffF(i-1)*(1/xH)^2 + xH*DiffF(i)*(-1/xH)^2;
    DT(i,i)     = xH*DiffT(i-1)*(1/xH)^2 + xH*DiffT(i)*(-1/xH)^2;
   
    M(i,i)      = xH/3 + xH/3;

    FissF(i,i)  = FF(i-1)*xH/3 + FF(i)*xH/3;
 
    FissT(i,i)  = FT(i-1)*xH/3 + FT(i)*xH/3;
    RemovF(i,i) = RF(i-1)*xH/3 + RF(i)*xH/3;
    RemovT(i,i) = RT(i-1)*xH/3 + RT(i)*xH/3;
   
    UpS(i,i)    = UpScat(i-1)*xH/3+ UpScat(i)*xH/3;
    DownS(i,i)  = DownScat(i-1)*xH/3 + DownScat(i)*xH/3;
    for j=2:N
    if abs(i-j)<2 && i ~= j
      DF(i,j)     = DiffF(min(i,j))*(1/xH * -1/xH)*xH ;
      DT(i,j)     = DiffT(min(i,j))*(1/xH * -1/xH)*xH ;
     
      M(i,j)      =  xH/6;  
         
      FissF(i,j)  =  FF(min(i,j))*xH/6;
      FissT(i,j)  =  FT(min(i,j))*xH/6;
      RemovF(i,j) =  RF(min(i,j))*xH/6;
      RemovT(i,j) =  RT(min(i,j))*xH/6;
     
      UpS(i,j)    = UpScat(min(i,j))*xH/6;
      DownS(i,j)  = DownScat(min(i,j))*xH/6;
    end
   end
end
D = [DF, zeros(N+1,N+1);zeros(N+1,N+1),DT]; %Creating two group by blocks
D = D(any(D,2),:); %trimming zeros
D = D(:,any(D,1));

M = [M, zeros(N+1,N+1);zeros(N+1,N+1),M];
M = M(any(M,2),:); %trimming zeros
M = M(:,any(M,1));

F            = [FissF, FissT;zeros(N+1,N+1),zeros(N+1,N+1)];
F(:,2*(N+1)) = [];
F(:,1)       = [];
F(2*(N+1),:) = [];
F(1,:)       = [];
F(:,N+2)     = [];
F(:,N+1)     = [];
F(N+2,:)     = [];
F(N+1,:)     = [];

S = [-RemovF, UpS; DownS,-RemovT];
S = S(any(S,2),:); %trimming zeros
S = S(:,any(S,1));

V = zeros(2*(N-1),2*(N-1));   %velocity matrix

for i=1:N-1
    V(i,i)         =   1.7869*10^9; %Fast in cm/s
    V(i+N-1,i+N-1) =   4.1896*10^7; %Thermal in cm/s
end
alphasoln      = zeros(2*(N+1),1);          
alphatotal     = zeros(2*N+2,TimeSteps+1);
for i=1:N+1
    alphasoln(i)     = f(x(i));
    alphasoln(i+N+1) = f(x(i));
end
alphasoln(1) = 0;
alphasoln(N+1) = 0;
alphasoln(N+2) = 0;
alphasoln(2*(N+1)) = 0;

u0 = alphasoln;

alphasoln = alphasoln(any(alphasoln,2),:); %trimming zeros
alphasoln = alphasoln(:,any(alphasoln,1));
 
P_unscaled    = (F*alphasoln)*200*1.6022E-13;
Scalingfactor = 40E3/sum(P_unscaled);            %Scaled to 40kW/cm^3

alphasoln     = alphasoln*Scalingfactor;
u0            = u0*Scalingfactor;

alphatotal(2:N,1)       = alphasoln(1:N-1);      
alphatotal(N+3:2*N+1,1) = alphasoln(N:2*N-2);  

%Power iterate to find k init
alphaguess    = zeros(2*(N-1),1);
for i=1:length(alphaguess)
    alphaguess(i) = rand;
end
invLHS   = (D-S)\eye(2*(N-1));
keff     = 1;
k_tot    = zeros;
k_tot(1) = keff;
Q        = zeros(2*(N-1),1);
P        = zeros(2*(N-1),1);
error=1;
i=2;
while 1>=0
if error<10^-8
  break
end
Q    = 1/keff*F*alphaguess;
L    = norm(Q(:,1),1);      %total fissions using old phi
alphaguess  = invLHS*Q;     %finding new phi
P    = F*alphaguess;
P    = norm(P(:,1),1);      %total fissions using new phi
keff = P/L;
k_tot(i) = keff;
error = abs(k_tot(i)-k_tot(i-1));
i=i+1;
end

trim      = zeros(2*(N+1),(TimeSteps-1)/print+1);
trim(:,1) = alphatotal(:,1);

B   = M\eye(2*(N-1)); %inverse of mass matrix
RHS = B*V*(-D+S+1/keff*F);
mat = (eye(2*(N-1))-0.5*tH*RHS)\(eye(2*(N-1))+0.5*tH*RHS);
for i=1:TimeSteps %trapezoidal method for alpha
alphasoln                 = mat*alphasoln;
alphatotal(2:N,i+1)       = alphasoln(1:N-1);
alphatotal(N+3:2*N+1,i+1) = alphasoln(N:2*N-2);  
k = int32(mod(i,print));
if k==0
   trim(2:N,(i-1)/print+1)       = alphatotal(2:N,i+1);
   trim(N+3:2*N+1,(i-1)/print+1) = alphatotal(N+3:2*N+1,i+1);
end   
end
if print==1
    trim = [alphatotal(:,1) trim];
end

if enrich ==1
  S1 = trim;
elseif enrich == 2
  S2 = trim;
elseif enrich == 3
  S3 = trim;
elseif enrich == 4
  S4 = trim;
  writematrix(trim, 'FOM_4.5.txt'); %writing FOM for 4.5 case to show POD dynamics
  writematrix(u0, 'u0_4.5.txt');
  writematrix(RHS, 'RHS_4.5.txt');
elseif enrich == 5
  S5 = trim;
elseif enrich==6
writematrix(trim, 'FOM_3.7.csv')
end
end

M_param_30 = zeros(2,TimeSteps/print+1);
M_param_35 = zeros(2,TimeSteps/print+1);
M_param_40 = zeros(2,TimeSteps/print+1);
M_param_45 = zeros(2,TimeSteps/print+1);
M_param_50 = zeros(2,TimeSteps/print+1);
M_param_37 = zeros(2,TimeSteps/print+1);

for i=0:TimeSteps/print
    M_param_30(:,i+1) = [3.0 , vt(i*print+1)].';
    M_param_35(:,i+1) = [3.5 , vt(i*print+1)].';
    M_param_40(:,i+1) = [4.0 , vt(i*print+1)].';
    M_param_45(:,i+1) = [4.5 , vt(i*print+1)].';
    M_param_50(:,i+1) = [5.0 , vt(i*print+1)].';
    M_param_37(:,i+1) = [3.7 , vt(i*print+1)].';
end
Snapshot = [S1 S2 S3 S4 S5];
M_param  = [M_param_30 M_param_35 M_param_40 M_param_45 M_param_50];

basis = 64;   %Number of orthogonal modes to take

G = zeros(snapshots*(TimeSteps/print+1),basis+10);
for i=1:snapshots*(TimeSteps/print+1)
   for j=1:basis+10
     G(i,j) = normrnd(0,1);
   end
end

Y       = (Snapshot*Snapshot.')*Snapshot*G; %Rand subspace of FOM
[Q1,R1] = qr(Y);
Q1      = Q1(:,1:basis);
B       = Q1.'*Snapshot;

[Uy,Sy,Vy] = svd(B);  %SVD of subspace of snapshot matrix
UN         = Q1*Uy;

size     = size(Snapshot);
Permut   = randperm(size(2));
Shuffled_S = zeros(size(1),size(2));
Shuffled_M = zeros(2, size(2));
for i=1:size(2)
    Shuffled_S(:,i) = Snapshot(:,Permut(i));  %Shuffling snapshot matrix
    Shuffled_M(:,i) = M_param(:,Permut(i));   %Shuffling parameter matrix
end

S_train = UN.'*Shuffled_S;  %Training data for CAE
S_train = S_train.';      %making time in rows for keras Conv_1D
Shuffled_M = Shuffled_M.';

max = 5.808805617696160E14;
min = -1.836615412714180E15;

for i=1:15005
    for j=1:64
        S_train(i,j) = (S_train(i,j)-min)/(max-min);
    end 
end

writematrix(UN, 'U_N.csv'); %Singular vectors in space
writematrix(S_train, 'S_train_normalized.csv');  %Shuffled data
writematrix(Shuffled_M, 'M.csv');
writematrix(M_param_37, 'M_37.csv');

%{
figure()
hold on
plot(diag(Sy)/sum(diag(Sy)),'ko')
hold off

figure()
hold on
plot(x,UN(1:N+1,1))
plot(x,UN(1:N+1,2))
legend('1','2')
title('First two sing vectors for fast flux snapshot matrix')
hold off
%}
