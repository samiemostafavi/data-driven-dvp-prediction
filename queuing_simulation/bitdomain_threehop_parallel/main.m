%% Prep the queues

% format shortG
restoredefaultpath

clear all;
close all;

SIM_NUM = '1';
NUM_WORKERS = 2;
initial_transient_proportion = 0.1; % percentage

sim_name = 'sim_three_hop';
sim_vars = [ 100000, 25  ...            % (1)  arrival_duration (T_max)         (2)  arrivalrate (rho)    
             1e-3,  ...                 % (3)  timslots duration (seconds)
             20e3, 5, ...               % (4)  hop 1 bandwidth in Hz        (5)  hop 1 SNR in db
             20e3, 5, ...               % (6)  hop 2 bandwidth in Hz        (7)  hop 2 SNR in db
             20e3, 5, ...               % (8)  hop 3 bandwidth in Hz        (9)  hop 3 SNR in db
             inf, inf, inf  ...         % (10)  queuecapacity_uplink     (11) queuecapacity_compute  (12) queuecapacity_downlink
             rand(1,1)*100000 ];        % (13) randomseed_offset 

         
%% Dataset Generation

tic

stop_time = '100000'; %'50000',ml2,arrival 0.8 -> 292 sec, not that accurate

numSims = 2;
simIn(1:numSims) = Simulink.SimulationInput(sim_name);
seedsOffsets = floor(rand(numSims,1)*100000);
for idx = 1:numSims
    sim_vars(13) = seedsOffsets(idx);
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
recordsTable = table;
virtualDelay = [];
for n = 1:numSims
    recordsTable = [recordsTable; logs2table(simOut(n).logsout,initial_transient_proportion)];
    at = get(simOut(n).logsout,'cumulative_arrival').Values.Data;
    dt = get(simOut(n).logsout,'cumulative_departure').Values.Data;
    virtualDelay = [virtualDelay, virtualdelay(at,dt)];
end

tail = calctail([0:10],virtualDelay);
createtailfigure([0:10],tail);

toc

clear 'simIn' 'simOut' 'idx' 'n';

%% Save all usefull data to 2 files: a .mat and a .parquet

% create the folder if does not exist
if not(isfolder('saves/'))
    mkdir('saves/');
end

clk_str = strrep(strrep(strrep(datestr(clock),' ','_'),':','_'),'-','_');
filename_meta = 'sim3hop_'+sprintf("%s",SIM_NUM)+'_metadata'+'_'+clk_str;
filename_dataset = 'sim3hop_'+sprintf("%s",SIM_NUM)+'_dataset'+'_'+clk_str;

save('saves/'+filename_meta+'.mat','sim_name','sim_vars','stop_time','initial_transient_proportion',  ...
            'numSims','SIM_NUM','NUM_WORKERS','clk_str','seedsOffsets');
        
parquetwrite('saves/'+filename_dataset+'.parquet',recordsTable);

