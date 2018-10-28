function [min_aic, min_a, min_b, min_c, aic_arr] = rels_search_abc(y, u, range_na, range_nb, range_nc, rho)
    min_aic = inf;
    for n_a = range_na
        for n_b = range_nb
            for n_c = range_nc
                [theta, aic] = relsf(y,u,n_a,n_b,n_c,0,rho);
                aic_arr(n_a,n_b,n_c) = aic;
                if(aic < min_aic)
                    min_aic = aic;
                    min_a = n_a;
                    min_b = n_b;
                    min_c = n_c;
                end
            end
        end
    end
end