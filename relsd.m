function [theta, P, xi, thetas] = relsd(y, u, n_a, n_b, n_c, d, rho)
    % Recursive extended least square 
    % CARMA model: A(q^{-1}) y(k) = B(q^{-1}) u(k) + C(q^{-1}) \xi(k)
    % "extended" data: $\hat{x}(k) = (-y(k-1),\dots,-y_{k-n_a},u(k),\dots,u(k-n_b),\dots,\xi(k-1),\dots,\xi(k-n_c))$
    
    num_p = n_a + n_b + 1 - d + n_c;

    theta = 1e-3 * ones(num_p, 1);
    P = 1e6 * eye(num_p);
    xi = y;
    
    for k = max([n_a n_b n_c])+1:size(u,1)
        x = [-y(k-1:-1:k-n_a) ; u(k-d:-1:k-n_b) ; xi(k-1:-1:k-n_c)];
        K = P * x /(rho + x' * P * x);
        theta = theta + K * (y(k) - x' * theta);
        thetas(:,k) = theta;
        P = 1/rho * (eye(num_p) - K * x') * P;
        xi(k) = y(k) - x' * theta;
    end
end 