load('aa.mat');
net= dagnn.DagNN; 
iv_name1 = 'input';
convBlock = dagnn.Conv('size', [10,10,3,3], 'pad', [0,0,0,0],'stride', [1,1], 'hasBias', false);
net.addLayer('conv',convBlock, {iv_name1}, {'conv'},{'conv_params'}); 
iv_name2 = 'conv';
Block = dagnn.Crop('crop', [3,3]);
net.addLayer('crop',Block, {iv_name1,iv_name2}, {'crop'});%no params, the same as Sum

for i=1:3
    net.vars(i).precious=1;
end

net.initParams()

net.eval(inputs);
size(net.vars(3).value)