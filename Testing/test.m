clear

L         = 437.58;  %total length of core
n         = 8;       %Cells per assembly
Assem     = 34;      %Number of assemblies
N         = n*Assem; %Total cells for each energy group
xH        = L/N;     %Width of each mesh cell

StartX    = -L/2;
EndX      = L/2;
x         = StartX:xH:EndX;

UN = csvread('U_N.csv');
S_train = csvread('S_train_normalized.csv');
S_train = S_train(any(S_train,2),:); 
S_train = S_train(:,any(S_train,1));

max = -1.6918E15;
min = -1.574E7;

CNN = csvread('DFNN_test.csv');
for i=1:15005
   for j=1:20
       S_train(i,j) = S_train(i,j)*(max-min)+min;
   end
end
S_train = S_train.';
FOM = UN*S_train(:,1);

for i = 1:20
    CNN(i) = CNN(i)*(max-min)+min;
end
Estimate = UN*CNN;

diff = zeros(1,546);
for i=1:546
    diff(i) = abs(FOM(i)-Estimate(i))/FOM(i);
end

figure()
hold on
plot(x,FOM(1:N+1,1))
plot(x,Estimate(1:N+1,1))
legend('FOM','Autoencoder output')
title('Fast Flux')
hold off

figure()
hold on
plot(x,FOM(N+2:2*(N+1),1))
plot(x,Estimate(N+2:2*(N+1),1))
legend('FOM','Autoencoder output')
title('Thermal Flux')
hold off

figure()
hold on 
plot(x,diff(1:N+1))
title('Fast error')
ylim([0 0.5])
hold off
figure()
hold on 
plot(x,diff(N+2:2*(N+1)))
title('Thermal error')
ylim([0 0.5])
hold off
