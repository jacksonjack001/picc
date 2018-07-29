load('aa.mat');
% inputs(4)=[];inputs(3)=[];
net= dagnn.DagNN; %network

iv_name = 'input';
poolBlock = dagnn.Pooling('method', 'max', 'poolSize', [2 2], 'stride', 2);
net.addLayer('pool1',poolBlock, iv_name,'pool1',{});

n=numel(net.vars);
for i=1:n
    net.vars(i).precious=1;
end

net.initParams()

net.eval(inputs)

for i=1:n
    size(net.vars(i).value)
end

net.vars(1).value(1:4,1:4,1,1)
net.vars(2).value(1:2,1:2,1,1)

% size(net.params(1).value)
% size(net.params(2).value)

