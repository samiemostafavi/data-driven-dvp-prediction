function plotgraph(data,str)

    figure;
%     subplot(2,1,1);
%     h = cdfplot(data);
%     set(h,'Marker', '.');
%     grid on
%     xl = xlim;
%     title('CDF');
%     ylabel('probability');
%     xlabel('time');
%     subplot(2,1,2);
    h = histogram(data,'Normalization','pdf');
    xl = xlim;
    hold on
    xline(mean(data),'-',strcat("mean: ", string(mean(data))));
    hold on
    xline(prctile(data,99),'-',strcat("99 percentile: ", string(prctile(data,99))));
    grid on
    xlim([xl(1), xl(end)]);
    title(str);
    ylabel('probability');
    xlabel('time');

end

