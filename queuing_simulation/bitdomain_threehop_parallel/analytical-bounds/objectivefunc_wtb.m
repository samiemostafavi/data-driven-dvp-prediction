function result = objectivefunc_wtb(s,gamma,W,rho,sigma,x1,omega,t)
    part1 = exp(s*(x1+sigma-(rho*omega)));
    part2 = (vzero(s,gamma,W,rho)^omega)/(1-vzero(s,gamma,W,rho));
    part3 = ((vzero(s,gamma,W,rho)^t)*(1-vzero(s,gamma,W,rho)))+((vzero(s,gamma,W,rho)/exp(s*x1))*(1-(vzero(s,gamma,W,rho)^(t-1))));
    result = part1*part2*part3;
end

