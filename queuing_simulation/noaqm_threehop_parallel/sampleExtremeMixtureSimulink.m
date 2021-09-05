function result = sampleExtremeMixtureSimulink(threshold,gamma_scale,gamma_shape,gpd_k,rnd)


% making conditional distribution
% gamma + gpd

sample = rnd;
% threshold = 0.9;

if sample < threshold
    % gamma
    % gamma_scale = 1/2;
    % gamma_shape = 10;
    result = gaminv(sample,gamma_shape,gamma_scale);
else
    % gpd
    % gpd_k = 0.5;
    % gpd_sigma = 0.5;
    theta = gaminv(threshold,gamma_shape,gamma_scale);
    gpd_sigma = 1/gampdf(theta,gamma_shape,gamma_scale)*((1-threshold)^2);

    gpd_sample = (sample-threshold)/(1-threshold);
    result = gpinv(gpd_sample,gpd_k,gpd_sigma)/(1-threshold)+theta; 
end

