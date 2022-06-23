function res_struct = getRandFromState(state,seed,rsize)
    % state size is 625x1 uint32 for mt19937ar
    rs = RandStream('mt19937ar','Seed',seed);
    rs.State = state;
    % we have to put the result in an struct together with the new state
    res_struct.rand_num = rand(rs,rsize); % stream state changes when a number is generated
    res_struct.updated_state = rs.State;
end

