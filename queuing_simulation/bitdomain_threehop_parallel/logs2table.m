function recordsTable = logs2table(logsout,transient_prop)

    % Get the data

    endtimes = get(logsout,'endtime').Values.Data;
    ids = get(logsout,'id').Values.Data;
    starttimes = get(logsout,'starttime').Values.Data;
    queuetimes_uplink = get(logsout,'queuetime_uplink').Values.Data;
    servicetimes_uplink = get(logsout,'servicetime_uplink').Values.Data;
    queuetimes_compute = get(logsout,'queuetime_compute').Values.Data;
    servicetimes_compute = get(logsout,'servicetime_compute').Values.Data;
    queuetimes_downlink = get(logsout,'queuetime_downlink').Values.Data;
    servicetimes_downlink = get(logsout,'servicetime_downlink').Values.Data;
    
    h1_uplink_queuen = get(logsout,'queuen_uplink_bu').Values.Data;
    h1_uplink_servern = get(logsout,'servern_uplink_bu').Values.Data;
    h1_compute_queuen = get(logsout,'queuen_compute_bu').Values.Data;
    h1_compute_servern = get(logsout,'servern_compute_bu').Values.Data;
    h1_downlink_queuen = get(logsout,'queuen_downlink_bu').Values.Data;
    h1_downlink_servern = get(logsout,'servern_downlink_bu').Values.Data;
    
    h2_compute_queuen = get(logsout,'queuen_compute_bc').Values.Data;
    h2_compute_servern = get(logsout,'servern_compute_bc').Values.Data;
    h2_downlink_queuen = get(logsout,'queuen_downlink_bc').Values.Data;
    h2_downlink_servern = get(logsout,'servern_downlink_bc').Values.Data;
    
    h3_downlink_queuen = get(logsout,'queuen_downlink_bd').Values.Data;
    h3_downlink_servern = get(logsout,'servern_downlink_bd').Values.Data;

    % first, sort all the data according to ids
    sorteddata = sortrows([ids,starttimes, endtimes ...
                           queuetimes_uplink, servicetimes_uplink, queuetimes_compute, servicetimes_compute, queuetimes_downlink,servicetimes_downlink,  ...
                           h1_uplink_queuen, h1_uplink_servern, h1_compute_queuen, h1_compute_servern, h1_downlink_queuen, h1_downlink_servern,  ...
                                                                h2_compute_queuen, h2_compute_servern, h2_downlink_queuen, h2_downlink_servern,  ...
                                                                                                       h3_downlink_queuen, h3_downlink_servern ]);

    ids = sorteddata(:,1);
    starttimes = sorteddata(:,2);
    endtimes = sorteddata(:,3);
    
    queuetimes_uplink = sorteddata(:,4);
    servicetimes_uplink = sorteddata(:,5);
    queuetimes_compute = sorteddata(:,6);
    servicetimes_compute = sorteddata(:,7);
    queuetimes_downlink = sorteddata(:,8);
    servicetimes_downlink = sorteddata(:,9);
    
    h1_uplink_queuen = sorteddata(:,10);
    h1_uplink_servern = sorteddata(:,11);
    h1_compute_queuen = sorteddata(:,12);
    h1_compute_servern = sorteddata(:,13);
    h1_downlink_queuen = sorteddata(:,14);
    h1_downlink_servern = sorteddata(:,15);
    
    h2_compute_queuen = sorteddata(:,16);
    h2_compute_servern = sorteddata(:,17);
    h2_downlink_queuen = sorteddata(:,18);
    h2_downlink_servern = sorteddata(:,19);
    
    h3_downlink_queuen = sorteddata(:,20);
    h3_downlink_servern = sorteddata(:,21); 

    % second, name the delays
    interarrival = diff(starttimes);
    end2enddelay = endtimes - starttimes;

    queuedelay_uplink = queuetimes_uplink - starttimes;
    servicedelay_uplink = servicetimes_uplink - queuetimes_uplink;
    queuedelay_compute = queuetimes_compute - servicetimes_uplink;
    servicedelay_compute = servicetimes_compute - queuetimes_compute;
    queuedelay_downlink = queuetimes_downlink - servicetimes_compute;
    servicedelay_downlink = servicetimes_downlink - queuetimes_downlink;

    totaldelay_uplink = servicetimes_uplink - starttimes;
    totaldelay_compute = servicetimes_compute - servicetimes_uplink;
    totaldelay_downlink = endtimes - servicetimes_compute; 

    
    % third, record
    records = [end2enddelay, ... %1
              queuedelay_uplink, servicedelay_uplink, totaldelay_uplink, queuedelay_compute, servicedelay_compute, totaldelay_compute, queuedelay_downlink,servicedelay_downlink,totaldelay_downlink, ... %2,3,4,5,6,7,8,9,10
              h1_uplink_queuen+h1_uplink_servern, h1_compute_queuen+h1_compute_servern, h1_downlink_queuen+h1_downlink_servern,  ... %11-13
                                                  h2_compute_queuen+h2_compute_servern, h2_downlink_queuen+h2_downlink_servern,  ... %14-15
                                                                                        h3_downlink_queuen+h3_downlink_servern ];    %16

    % remove the initial transient samples   
    records = records(transient_prop*size(records,1)+1:end,:);
    
    recordsTable = array2table(records,'VariableNames',{'end2enddelay', ...
                                                        'queuedelay_uplink','servicedelay_uplink','totaldelay_uplink','queuedelay_compute','servicedelay_compute','totaldelay_compute','queuedelay_downlink','servicedelay_downlink','totaldelay_downlink', ...
                                                        'h1_uplink_netstate','h1_compute_netstate','h1_downlink_netstate',...
                                                                             'h2_compute_netstate','h2_downlink_netstate',...
                                                                                                   'h3_downlink_netstate'});
    
end

