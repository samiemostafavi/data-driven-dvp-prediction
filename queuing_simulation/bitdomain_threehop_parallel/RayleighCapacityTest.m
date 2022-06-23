%% Formulation from the paper: 
% Transient Delay Bounds for Multi-Hop Wireless Networks
% Jaya Prakash Champati, Hussein Al-Zubaidy, James Gross

% for 1ms slot duration, 20kHz bandwidth, and 5 db snr the average service rate amounts to about 34 bits per time slot 

w = 20e3; % channel bandwidth in hz
time_slot_duration = 0.001; % in seconds
len = 1e2; % total number of simulation timeslots

snr = 0; % in db
capacity = zeros(1,len);
for i=[1:len]
    capacity(i) = time_slot_duration*w*log2(1+exprnd(1)*(10^(snr/10)));
end
figure;
plot(cumsum(capacity))

snr = 5; % in db
capacity = zeros(1,len);
for i=[1:len]
    capacity(i) = time_slot_duration*w*log2(1+exprnd(1)*(10^(snr/10)));
end

hold on
plot(cumsum(capacity))
title('Cumulative service')
xlabel('timestep')
ylabel('bits')