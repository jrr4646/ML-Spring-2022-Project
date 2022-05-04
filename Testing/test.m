clear

L         = 437.58;  %total length of core
n         = 8;       %Cells per assembly
Assem     = 34;      %Number of assemblies
N         = n*Assem; %Total cells for each energy group
xH        = L/N;     %Width of each mesh cell

StartX    = -L/2;
EndX      = L/2;
x         = StartX:xH:EndX;

test_t = linspace(0, 3000, 11);

UN = csvread('U_N.csv');
S_train = csvread('S_train_normalized_64.csv');
S_train = S_train.';
S_train = S_train(:,1:11); %First 11 random snapshots

%max = -1.6918E15;
%min = -1.574E7;

max = 5.808805617696160E14;
min = -1.836615412714180E15;

CNN = csvread('train.csv');
CNN = CNN.';

prediction = csvread('test_3.7.csv');
prediction = prediction.';
FOM_test = csvread('FOM_3.7.csv');

for i=1:64
   for j=1:11
       S_train(i,j) = S_train(i,j)*(max-min)+min;
       prediction(i,j) = prediction(i,j)*(max-min)+min;
       CNN(i,j) = CNN(i,j)*(max-min)+min;
   end
end
FOM_train = UN*S_train;
Est_train = UN*CNN;       %11 random snapshots of train data
Est_test  = UN*prediction; %3.7% enrichment at t=3e-5
%{
figure()
hold on
title('Thermal Flux over 11 random snapshots')
ylim([0 3e13])
for i=1:11
  cla
  plot(x,FOM_train(N+2:2*(N+1),i))
  plot(x,Est_train(N+2:2*(N+1),i))
  legend('FOM','Autoencoder output')
  pause(1.5)
end
hold off

figure()
hold on
title('Thermal Flux for 3.7% enrich for 0-3e-5')
ylim([0 3e13])
for i=1:11
  cla
  plot(x,FOM_test(N+2:2*(N+1),test_t(i)+1))
  plot(x,Est_test(N+2:2*(N+1),i))
  legend('FOM','Autoencoder output')
  pause(1.5)
end
hold off
%}

v = VideoWriter('test_3.7.avi');
open(v);
for i=1:11
  hold on
  cla
  ylim([0 2.6e13])
  plot(x,FOM_test(N+2:2*(N+1),test_t(i)+1))
  plot(x,Est_test(N+2:2*(N+1),i))
  pause(1)
  frame = getframe();
  writeVideo(v,frame);
  hold off
end
close(v);
