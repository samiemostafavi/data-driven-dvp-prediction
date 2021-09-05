function dt = getArrivalTime(arrivalfunction,arrivalrate,randomseed)

    coder.extrinsic('initRandStream')
    coder.extrinsic('getRandFromState')
    
    % define 
    persistent rs_state;
    if isempty(rs_state)
        % always for this type 625x1 uint32
        % init the result struct to handle mxArray
        rs_state = zeros([625,1],'uint32');
        rs_state = initRandStream(12345+randomseed);
    end
    
    if arrivalfunction == 0
        dt = 1/arrivalrate;
    elseif arrivalfunction == 1

        % init the result struct to handle mxArray
        rand_struct.rand_num = 0.0;
        rand_struct.updated_state = zeros([625,1],'uint32');
        % get the random number and update the random stream state
        rand_struct = getRandFromState(rs_state,12345+randomseed,1);
        rs_state = rand_struct.updated_state;
        rnd = rand_struct.rand_num;

        % random dt calculations
        mean = 1/arrivalrate;
        dt = -mean*log(1-rnd);
    else
        dt = 1/arrivalrate;
    end
  
end