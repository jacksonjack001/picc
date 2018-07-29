load('aa.mat');
net= dagnn.DagNN; %network

iv_name = 'input';
convBlock1 = dagnn.Conv('size', [3,3,3,12], 'pad', [1,1,1,1],'stride', [1,1], 'hasBias', false);
net.addLayer('pre_conv1',convBlock1,iv_name,{'pre_conv1'},{'conv_pre_params1'});

net.addLayer('pre_conv2',convBlock1,'pre_conv1',{'pre_conv2'},{'conv_pre_params2'});

net.addLayer('pre_conv3',convBlock1,'pre_conv2',{'pre_conv3'},{'conv_pre_params3'});

net.addLayer('join', dagnn.Concat('dim',3), {'pre_conv1','pre_conv2','pre_conv3'}, {'join'});

for i=1:5
    net.vars(i).precious=1;
end
conv=dagnn.Conv('size',[3,3,36,12], 'pad', [1,1,1,1],'stride', [1,1], 'hasBias', false);
net.addLayer('join_1',conv,{'join'},{'join_1'},{'join1_params1'});
net.initParams()


for i=1:5
    size(net.vars(i).value)
end

size(net.params(1).value)
size(net.params(2).value)

net.eval(inputs)