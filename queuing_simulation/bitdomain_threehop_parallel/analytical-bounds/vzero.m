function result = vzero(s,gamma,W,rho)
    result = exp(s*rho)*exp(1/gamma)*(gamma^((-s*W)/(log(2))))*gamma_incomplete((1-((s*W)/(log(2)))),(1/gamma));
end

