function plot_trace(trace, idxs, labels)
    figure
    plot(trace(idxs(1),:));
    hold on
    for k = 2:length(idxs)
        plot(trace(idxs(k),:));
    end
    hold off
    legend(labels);
end