function result = wtb(smax_,t_,omega_,gamma_,W_,x1_,sigma_,rho_)

    func = @(s) objectivefunc_wtb(s,gamma_,W_,rho_,sigma_,x1_,omega_,t_);
    options = optimset('TolX',1e-14,'MaxIter',10000);
    [xval,funcval] = fminbnd(func,0,smax_,options);
    result = min(1,funcval);
end

