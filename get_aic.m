function y = aic(xi, k, theta, ro)
    mu = mean(xi);
    sd = std(xi);
    logprob = -sum(((xi - mu)/sd).^2/2) - log(sqrt(2*pi)*sd)*size(xi,1);
    y = 2*k - logprob;
end