function result = objectivefunc_sta(s,gamma,W,rho,sigma,x1,omega)
    part1 = exp(s*(x1+sigma-(rho*omega)))/(1-vzero(s,gamma,W,rho));
    part2 = min(1,((vzero(s,gamma,W,rho))^(omega)));
    result = part1*part2;
end

