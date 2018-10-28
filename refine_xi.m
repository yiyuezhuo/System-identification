function xi = refine_xi(y, u, n_a, n_b, n_c, d, rho, theta)
    % shit Matlab
    num_p = n_a + n_b + 1 - d + n_c;

    %xi = y;
    xi = zeros(size(y));
    
    for k = max([n_a n_b n_c])+1:size(u,1)
        x = [-y(k-1:-1:k-n_a) ; u(k-d:-1:k-n_b) ; xi(k-1:-1:k-n_c)];
        xi(k) = y(k) - x' * theta;
    end
end