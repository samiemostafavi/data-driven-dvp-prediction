%% Prep the queues

% format shortG
restoredefaultpath

clear all;
close all;

SIM_NUM = '1';
NUM_WORKERS = 2;
initial_transient_proportion = 0.1; % percentage

sim_name = 'sim_three_hop';
sim_vars = [ 0, 0.9  ...                % (1)  arrivalfunction          (2)  arrivalrate  
             8, 1, ...                  % (3)  servicefunction_uplink   (4)  servicerate_uplink
             8, 1, ...                  % (5)  servicefunction_compute  (6)  servicerate_compute
             8, 1, ...                  % (7)  servicefunction_downlink (8)  servicerate_downlink
             inf, inf, inf  ...         % (9)  queuecapacity_uplink     (10) queuecapacity_compute  (11) queuecapacity_downlink
             rand(1,1)*100000 ];        % (12) randomseed_offset 

         
%% Dataset Generation

tic

stop_time = '12222'; %'50000',ml2,arrival 0.8 -> 292 sec, not that accurate

numSims = 2;
simIn(1:numSims) = Simulink.SimulationInput(sim_name);
seedsOffsets = floor(rand(numSims,1)*100000);
for idx = 1:numSims
    sim_vars(12) = seedsOffsets(idx);
    simIn(idx) = simIn(idx).setVariable('sim_vars', sim_vars);
    simIn(idx) = simIn(idx).setModelParameter('StopTime',stop_time); 
end
toc

tic
warning off;

delete(gcp('nocreate'))         % shutdown the parallel pool
parpool('local',NUM_WORKERS);   % start a new one

% Simulate the model
simOut = parsim(simIn,'UseFastRestart','on'); % ,'ShowProgress', 'on', 'ShowSimulationManager','on'
records = [];
for n = 1:numSims
    records = [records; logs2record(simOut(n).logsout,initial_transient_proportion)];
end
toc

clear 'simIn' 'simOut' 'idx' 'n';

%% Save all usefull data to a file

% create the folder if does not exist
if not(isfolder('saves/'))
    mkdir('saves/');
end

filename = 'threehop_dataset_'+sprintf("%s",SIM_NUM)+'_'+strrep(strrep(strrep(datestr(clock),' ','_'),':','_'),'-','_')+'.mat';

save('saves/'+filename,'sim_name','sim_vars','stop_time','initial_transient_proportion',  ...
            'records','numSims','SIM_NUM','NUM_WORKERS','seedsOffsets','-v7.3');

