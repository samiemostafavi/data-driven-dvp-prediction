%clear all

% SNR
gamma = 3.16; % 5 db in linear
%gamma = 10; % 10 db in linear

% bandwidth
W = 20; % in kHz

t = 1;
%omega = 0.010; % end-to-end delay bound

% initial backlog
x1 = 40; 

% arrival parameters
sigma = 25;
rho = 0;
%smax = log(2)/W;
smax = 2;

arr = [];
for omega = [0:1:14]
    arr = [arr, wtb(smax,t,omega,gamma,W,x1,sigma,rho)];
    %arr = [arr, stationaryb(smax,omega,gamma,W,x1,sigma,rho)];
    %stationary bound is broken...
end

figure;
semilogy([0:1:14],arr,'Marker','*');
%semilogy([0:1:14],arr,'Marker','*');
ylim([1e-6,1]);
xlim([0,14]);