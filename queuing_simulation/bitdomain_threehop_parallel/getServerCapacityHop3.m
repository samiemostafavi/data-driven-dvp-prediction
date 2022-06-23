function capacity = getServerCapacityHop3(mean,time_slot_duration,snr,w,randomseed)
    
    coder.extrinsic('initRandStream')
    coder.extrinsic('getRandFromState')


    persistent rs_state;
    if isempty(rs_state)
        % always for this type 625x1 uint32
        % init the result struct to handle mxArray
        rs_state = zeros([625,1],'uint32');
        rs_state = initRandStream(83736+randomseed);
    end
    
    % init the result struct to handle mxArray
    rand_struct.rand_num = 0.0;
    rand_struct.updated_state = zeros([625,1],'uint32');
    % get the random number and update the random stream state
    rand_struct = getRandFromState(rs_state,83736+randomseed,1);
    rs_state = rand_struct.updated_state;
    rnd = rand_struct.rand_num;
    
    exponential_rnd = -mean*log(1-rnd);
    capacity = floor(time_slot_duration*w*log2(1+exponential_rnd*(10^(snr/10))));
    %disp('service capacity:')
    %disp(capacity)
end