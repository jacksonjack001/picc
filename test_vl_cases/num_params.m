% 1,3443,2597
% net=denseNet();
num=0;P=net.params;
for i=1:length(P)
    num=num+prod(size(P(i).value));
    prod(size(P(i).value))
end
num


N=[];P=net.params;
for i=1:length(P)
    if ~isempty(findstr('bn_x',P(i).name))
        P(i).name
        temp=prod(size(P(i).value))/2;
        N=[N;temp;temp];
        continue;
    end
    N=[N;prod(size(P(i).value))];
end
sum(N)



feature1 = squeeze(net.vars(net.getVarIndex('input')).value)