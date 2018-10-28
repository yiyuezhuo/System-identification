function [min_aic, min_a, min_b, min_c, min_d, aic_arr] = rels_search_abc(y, u, range_na, range_nb, range_nc, range_nd, rho)
    min_aic = inf;
    for n_a = range_na
        for n_b = range_nb
            for n_c = range_nc
                for n_d = range_nd
                    [theta, aic] = relsf(y,u,n_a,n_b,n_c,n_d,rho);
                    aic_arr(n_a,n_b,n_c,n_d) = aic;
                    if(aic < min_aic)
                        min_aic = aic;
                        min_a = n_a;
                        min_b = n_b;
                        min_c = n_c;
                        min_d = n_d;
                    end
                end
            end
        end
    end
end