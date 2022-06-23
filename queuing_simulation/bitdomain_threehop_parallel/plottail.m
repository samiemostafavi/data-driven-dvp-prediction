function  plottail(records,x,x1,ylimit,ylogscale,xlogscale)
    
    %x = [0:1:14];
    %ylimit = [1e-6, 1e0];
    %ylogscale = 1;
    %xlogscale = 0;
    %x1 = 0;

    records = records(records(:,2:end)==x1,1);

    res = [];
    for i=1:length(x)
        xi = x(i);
        res = [res, length(records(records>xi))/length(records)];
    end

    figure;
    plot(x,res,'-*')
    ylim(ylimit)
    xlim([x(1),x(end)])
    if ylogscale
        set(gca,'YScale','log');
    end
    if xlogscale
        set(gca,'XScale','log');
    end

end

