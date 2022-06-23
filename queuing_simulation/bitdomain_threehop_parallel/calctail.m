function tail = calctail(x,y)
    tail = [];
    for i=1:length(x)
        tail = [tail, length(y(y>=x(i)))/length(y)];       
    end
end

