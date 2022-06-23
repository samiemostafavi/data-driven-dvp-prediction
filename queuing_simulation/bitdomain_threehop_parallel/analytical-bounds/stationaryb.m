function result = stationaryb(smax_,omega_,gamma_,W_,x1_,sigma_,rho_)

    func = @(s) objectivefunc_sta(s,gamma_,W_,rho_,sigma_,x1_,omega_);
    options = optimset('TolX',1e-50,'MaxIter',10000);
    [xval,funcval] = fminbnd(func,0,smax_,options);
    result = min(1,funcval);
end

