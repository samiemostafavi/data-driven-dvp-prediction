function state = initRandStream(seed)
    % random stream state size is 625x1 uint32 for mt19937ar
    rs = RandStream('mt19937ar','Seed',seed);
    state = rs.State;
end

