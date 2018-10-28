function [theta, aic, thetas, xi, P] = relsf(y, u, n_a, n_b, n_c, d, rho)
    % This function tend to return all information you need.
    % The meaning of n_b is changed to keep consistence with PPT and material.
    
    % num_p = n_a + n_b + 1 - d + n_c;
    num_p = n_a + n_b + 1 + n_c;

    % estimate parameters
    theta = 1e-3 * ones(num_p, 1);
    P = 1e6 * eye(num_p);
    xi = y;
    
    for k = max([n_a d+n_b n_c])+1:size(u,1)
        % x = [-y(k-1:-1:k-n_a) ; u(k-d:-1:k-n_b) ; xi(k-1:-1:k-n_c)];
        x = [-y(k-1:-1:k-n_a) ; u(k-d:-1:k-d-n_b) ; xi(k-1:-1:k-n_c)];
        K = P * x /(rho + x' * P * x);
        theta = theta + K * (y(k) - x' * theta);
        thetas(:,k) = theta;
        P = 1/rho * (eye(num_p) - K * x') * P;
        xi(k) = y(k) - x' * theta;
    end

    % refine xi
    
    xi = zeros(size(y));
    
    for k = max([n_a d+n_b n_c])+1:size(u,1)
        x = [-y(k-1:-1:k-n_a) ; u(k-d:-1:k-d-n_b) ; xi(k-1:-1:k-n_c)];
        xi(k) = y(k) - x' * theta;
    end
    
    % compute AIC
    k = n_a+n_b+1+n_c;
    mu = mean(xi);
    sd = std(xi);
    logprob = -sum(((xi - mu)/sd).^2/2) - log(sqrt(2*pi)*sd)*size(xi,1);
    aic = 2*k - logprob;
    
end
