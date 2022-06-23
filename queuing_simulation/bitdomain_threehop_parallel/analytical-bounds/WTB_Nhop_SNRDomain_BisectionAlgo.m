%tic
clear all;
N = 1;
d = 0; %Delayed backlog
T = 1;
t = T + d;
totalInitialBacklog = 0; %100
x = totalInitialBacklog/N*ones(1,N);
snrdB = 5;
snr = 10^(snrdB/10); %Average SNR
W = 20;
ARRIVAL_RATE = 25;

numSlots = 15;
ViolationProb = zeros(1,numSlots);
A = ones(1,t+1);
%In Matlab the indices start from 1. Therefore, I use t + 1 for sequence A; In the
%paper the slots start from 0, so A(1) represents the arrivals in slot 0
%which is burst arrival.

if t == 1
    %sigma = x + 20;
    sigma = 25;
    rho = 0;
    A(2) = exp(sigma);
else
    sigma = 0;
    rho = ARRIVAL_RATE;
    for i = 2:t+1
        A(i) = exp((i-1)*rho);
    end
end

w = 0;

fileID = fopen('data.txt','w');

for i = 1:length(ViolationProb)
    tau = t + w;
    minProb = realmax;
    
    tic
    lb = 0.000001;
    ub = 1;

    while ub - lb > .0001
        midc = (ub + lb)/2;
        
        %Compute the value for midpoint between midc and lb
        s = (midc + lb)/2;
        V = exp(1/snr)*snr^(-s*W/log(2))*gamma_incomplete(1-W*s/log(2),1/snr);
        sum1 = 0;
        sum2 = 0;
       
        if N == 1
            sum1 = sum1 + A(t+1)^s*exp(s*x);
        else
            for u=0:N-2
                sum1 = sum1 + nchoosek(u+tau-1,tau-1)*A(t+1)^s*exp(s*sum(x(1:N-u)));
            end
            u = N - 1;
            sum1 = sum1 + nchoosek(u+tau-1,tau-1)*A(t+1)^s*exp(s*x(1));
        end
        
        if t > 1  %else sum2 is zero
            for u = 1:t-1
                sum2 = sum2 + A(t+1)^s/A(u+1)^s*V^(tau-u);
            end
        end        
        
        philb = sum1*V^tau + nchoosek(N+tau-2,tau-1)*sum2;
        sum1*V^tau
        nchoosek(N+tau-2,tau-1)*sum2
        
        %Compute the value for midpoint between midc and ub
        s = (midc + ub)/2;
        V = exp(1/snr)*snr^(-s*W/log(2))*gamma_incomplete(1-W*s/log(2),1/snr);
        sum1 = 0;
        sum2 = 0;
       
        if N == 1
            sum1 = sum1 + A(t+1)^s*exp(s*x);
        else        
            for u=0:N-2
                sum1 = sum1 + nchoosek(u+tau-1,tau-1)*A(t+1)^s*exp(s*sum(x(1:N-u)));
            end
            u = N - 1;
            sum1 = sum1 + nchoosek(u+tau-1,tau-1)*A(t+1)^s*exp(s*x(1));            
        end
        
        if t > 1  %else sum2 is zero
            for u = 1:t-1
                sum2 = sum2 + A(t+1)^s/A(u+1)^s*V^(tau-u);
            end
        end        
        
        phiub = sum1*V^tau + nchoosek(N+tau-2,tau-1)*sum2;        
       
        %Compare
       if(phiub > philb)
           ub = midc;
           minProb = philb;
       else
           lb = midc;
           philb = phiub;
       end  
    end
    toc
    
    %minProb
    %mins
    ViolationProb(i) = min(1,minProb);
    w = w+1;
end

%ViolationProb(7)
ViolationProb(10)
%toc

hold on; plot(0:length(ViolationProb)-1,ViolationProb,'-vb');

ylim([1e-6,1]);
xlim([0,14]);
set(gca, 'YScale', 'log')
