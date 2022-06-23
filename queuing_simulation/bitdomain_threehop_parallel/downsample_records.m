function new_recordstable = downsample_records(recordstable,x,maxlen,add_gnoise,sigma)

    %x = [0:10:100];
    %a = recordstable{:,11}==x;
    %b = logical(sum(a,2));
    %new_recordstable = recordstable(b,:);
    
    d = [];
    for i=1:length(x)
        xi = x(i);
        a = recordstable{:,11}==xi;
        b = find(a);
        if length(b)>maxlen
            c = randperm(length(b))';
            b = b(c(1:maxlen));
        end
        d = [d;b];
    end
    new_recordstable = recordstable(d,:);
    
    if add_gnoise
        myrands = normrnd(0,sigma,size(new_recordstable{:,1}));
        new_cul = new_recordstable{:,1} + myrands;
        new_recordstable{:,1} =  new_cul;
    end
    
end

