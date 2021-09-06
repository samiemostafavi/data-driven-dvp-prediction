

NUM_WORKERS = 18;
initial_transient_proportion = 0.1; % percentage

sim_name = 'sim_three_hop';
sim_vars = [ 0, 0.9  ...                % (1)  arrivalfunction          (2)  arrivalrate  
             8, 1, ...                  % (3)  servicefunction_uplink   (4)  servicerate_uplink
             8, 1, ...                  % (5)  servicefunction_compute  (6)  servicerate_compute
             8, 1, ...                  % (7)  servicefunction_downlink (8)  servicerate_downlink
             inf, inf, inf  ...         % (9)  queuecapacity_uplink     (10) queuecapacity_compute  (11) queuecapacity_downlink
             rand(1,1)*100000 ];        % (12) randomseed_offset 

Stoptime = 6000000; 

Results:
This configuration filled 32GB memory + 64GB swap entirely.
Created 87.479.865 samples
The simulation took 82 minutes

