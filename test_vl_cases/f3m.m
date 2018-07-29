function muX=f3m(X)
t=X;
for i=fliplr([1,2,4])
    t=mean(t,i);
end
muX=t;
end