function Wt = virtualdelay(At,Dt)
    Wt = [];
    for i=1:length(At)
        [val,id] = min(abs(Dt-At(i)));
        if val>At(i)
            id = id-1;
        end
        if id < i
            id = i;        
        end
        Wt = [Wt,id-i];
    end
end

