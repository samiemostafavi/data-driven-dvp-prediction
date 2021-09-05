function dt = getServiceTimeDownlink(servicefunction,servicerate,randomseed)
    
    % fix these
    %coder.extrinsic('lognrnd')
    %coder.extrinsic('gamrnd')
    
    coder.extrinsic('initRandStream')
    coder.extrinsic('getRandFromState')

    persistent olddt;
    if isempty(olddt)
        olddt = 0;
    end

    persistent rs_state;
    if isempty(rs_state)
        % always for this type 625x1 uint32
        % init the result struct to handle mxArray
        rs_state = zeros([625,1],'uint32');
        rs_state = initRandStream(72837+randomseed);
    end
    
    % init the result struct to handle mxArray
    rand_struct.rand_num = 0.0;
    rand_struct.updated_state = zeros([625,1],'uint32');
    % get the random number and update the random stream state
    rand_struct = getRandFromState(rs_state,72837+randomseed,1);
    rs_state = rand_struct.updated_state;
    rnd = rand_struct.rand_num;
    
    dt = 0;
    if servicefunction == 0
        dt = 1/servicerate;
    elseif servicefunction == 1
        mean = 1/servicerate;
        dt = -mean*log(1-rnd);
    elseif servicefunction == 2
        mean = 1/servicerate;
        newdt = -mean*log(1-rnd);
        correlation = 0.2;
        dt = olddt*correlation + (1-correlation)*newdt;
        olddt = dt;
    elseif servicefunction == 3
        mean = 1/servicerate;
        newdt = -mean*log(1-rnd);
        correlation = 0.5;
        dt = olddt*correlation + (1-correlation)*newdt;
        olddt = dt;
    elseif servicefunction == 4
        mean = 1/servicerate;
        newdt = -mean*log(1-rnd);
        correlation = 0.8;
        dt = olddt*correlation + (1-correlation)*newdt;
        olddt = dt;
    %%% Need to get fixed for a random seed %%%
    %elseif servicefunction == 5
    %    mean = 1/servicerate;
    %    shape = 5;
    %    scale = mean/shape;
    %    dt = gamrnd(shape,scale);
    %elseif servicefunction == 6
    %    mean = 1/servicerate;
    %    sigma = 1;
    %    mu = log(mean) - (sigma^2)/2;
    %    dt = lognrnd(mu,sigma);
    elseif servicefunction == 7
        mean = 1/servicerate;
        scale = mean/5;
        shape = 5;
        dt = sampleExtremeMixtureSimulink(0.8,scale,shape,0.1,rnd);
    elseif servicefunction == 8
        mean = 1/servicerate;
        scale = mean/5;
        shape = 5;
        dt = sampleExtremeMixtureSimulink(0.8,scale,shape,0.2,rnd);
    elseif servicefunction == 9
        mean = 1/servicerate;
        scale = mean/5;
        shape = 5;
        dt = sampleExtremeMixtureSimulink(0.8,scale,shape,0.4,rnd);
    else
        dt = 1/servicerate;
    end
    
    %disp('service dt:')
    %disp(dt)
end