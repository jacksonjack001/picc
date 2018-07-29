function net = unet_concat_BN_noCrop()
D=5;
N=2;
net= dagnn.DagNN;

fD = [64, 128, 256, 512, 1024]/2;
% conv�����ߣ�_conv����ұߣ�������reluconv1_2��relu_conv1_2
for d = 1:D
    deepest = d == D;
    net=contraction(net,d, deepest);
end
for d = D-1:-1:1
    deepest = d == (D-1);
    net=expansion(net, d, deepest);
end

% pred conv
iv_name='relu_conv1_2';
layername='prediction';ov_name='prediction';
params='prediction_params';
convBlock = dagnn.Conv('size', [1,1,fD(1),N], 'hasBias', false, 'pad',[0,0,0,0]);
net.addLayer(layername, convBlock, iv_name, ov_name, params);

%BN-final
iv_name=ov_name;
layername='bn_final';ov_name=layername;
ov_pre='bF_';params={[ov_pre,'bn_m'], [ov_pre,'bn_b'],[ov_pre,'bn_x']};
net.addLayer(layername, dagnn.BatchNorm('numChannels', convBlock.size(4)), iv_name, ov_name, params);

% pred relu   
% !!!!  wan e zhi yuan !!!!
% iv_name=ov_name;
% layername='softmax_fn';ov_name=layername;
% net.addLayer(layername, dagnn.SoftMax(), iv_name, ov_name,{});


% �������⣬label�ĳߴ���prediction�ĳߴ���ô˵?
iv_name={ov_name,'label'};
lossBlock=SegmentationLoss('loss', 'softmaxlog');
net.addLayer('objective', lossBlock, iv_name,{'objective'}) ;

% Error layer
iv_name={ov_name,'label'};
net.addLayer('accuracy',  SegmentationAccuracy(), iv_name, {'accuracy'}) ;

in=[73,74,62,53,44,35];
for i=1:length(in)
    net.vars(in(i)).precious=1;
end
    
net.initParams() ;

end


function net=contraction(net, d, deepest)
fD = [64, 128, 256, 512, 1024]/2;
% convBlock=dagnn.Conv('size', [3,3,0,0], 'hasBias', false);
%d1 conv1-bn1-relu1--conv2-bn2-relu2--pool
%d2 conv1-bn1-relu1--conv2-bn2-relu2--pool
%d5 ...

if d == 1
    incoming = 'input';
    convBlock1 = dagnn.Conv('size', [3,3,0,0], 'hasBias', false, 'pad',[1,1,1,1]);
    convBlock1.size(3:4) = [3, fD(d)];
    convBlock2 = dagnn.Conv('size', [3,3,0,0], 'hasBias', false, 'pad',[1,1,1,1]);
    convBlock2.size(3:4) = [fD(d), fD(d)];
else
    incoming = sprintf('pool%d', d-1);
    convBlock1 = dagnn.Conv('size', [3,3,0,0], 'hasBias', false, 'pad',[1,1,1,1]);
    convBlock1.size(3:4) = [fD(d-1), fD(d)];
    convBlock2 = dagnn.Conv('size', [3,3,0,0], 'hasBias', false, 'pad',[1,1,1,1]);
    convBlock2.size(3:4) = [fD(d), fD(d)];
end

%conv1
iv_name=incoming;
layername=sprintf('conv%d_1',d);ov_name=layername;
param=sprintf('conv%d_1_params',d);
net.addLayer(layername, convBlock1, iv_name, ov_name, param);

%BN1
iv_name=ov_name;
ov_pre=layername;layername=[ov_pre,'_bn'];ov_name=layername;
params={[ov_pre,'bn_m'], [ov_pre,'bn_b'],[ov_pre,'bn_x']};
net.addLayer(layername, dagnn.BatchNorm('numChannels', convBlock1.size(4)), iv_name, ov_name, params);

%relu1
iv_name=ov_name;
layername=sprintf('reluconv%d_1',d);ov_name=layername;
net.addLayer(layername, dagnn.ReLU(), iv_name, ov_name, {});

%conv2
iv_name=ov_name;
layername=sprintf('conv%d_2',d);ov_name=layername;
param=sprintf('conv%d_2_params',d);
net.addLayer(layername, convBlock2, iv_name, ov_name, param);

%BN2
iv_name=ov_name;
ov_pre=layername;layername=[ov_pre,'_bn'];ov_name=layername;
params={[ov_pre,'bn_m'], [ov_pre,'bn_b'],[ov_pre,'bn_x']};
net.addLayer(layername, dagnn.BatchNorm('numChannels', convBlock2.size(4)), iv_name, ov_name, params);

%relu2
iv_name=ov_name;
layername=sprintf('reluconv%d_2',d);ov_name=layername;
net.addLayer(layername, dagnn.ReLU(), iv_name, ov_name, {});

if ~deepest
    iv_name=ov_name;
    layername=sprintf('pool%d',d);ov_name=layername;
    poolBlock = dagnn.Pooling('method', 'max', 'poolSize', [2 2], 'stride', 2);
    net.addLayer(layername, poolBlock, iv_name, ov_name, {});
end

end

function net=expansion(net, d, deepest)
fD = [64, 128, 256, 512, 1024]/2;
% convBlock=dagnn.Conv('size', [3,3,0,0], 'hasBias', false);
%d4 convT��2--crop-concat--conv1-bn1-relu1--conv2-bn2-relu2
%d3 convT��2--crop-concat--conv1-bn1-relu1--conv2-bn2-relu2
%d1 ...

if deepest
    incoming = sprintf('reluconv%d_2',d+1);
else
    incoming = sprintf('relu_conv%d_2',d+1);
end

%����ϲ����, �ȷŴ��������������,Ҫ��֤һ�����Žӹ����ĳߴ�һ��
iv_name=incoming;
layername=sprintf('upconv%d',d);ov_name=layername;
param={sprintf('upconv%d_params',d)};
convBlockT = dagnn.ConvTranspose('size', [2,2, fD(d),fD(d+1)],'upsample',[2,2], 'hasBias', false);
net.addLayer(layername, convBlockT, iv_name, ov_name, param);

%��ϣ�concat
from_bridge = sprintf('reluconv%d_2',d);from_upconv=ov_name;
iv_name={from_bridge, from_upconv};
layername=sprintf('bridge_%d', d);ov_name=layername;
net.addLayer(layername, dagnn.Concat('dim',3), iv_name, ov_name,{});

%conv1
iv_name=ov_name;
layername=sprintf('d_conv%d_1',d);ov_name=layername;
param=sprintf('d_conv%d_1_params',d);
convBlock1 = dagnn.Conv('size', [3,3,fD(d)*2, fD(d)], 'hasBias', false, 'pad',[1,1,1,1]);
net.addLayer(layername, convBlock1, iv_name, ov_name, param);

%BN1
iv_name=ov_name;
ov_pre=layername;layername=[ov_pre,'_bn'];ov_name=layername;
params={[ov_pre,'bn_m'], [ov_pre,'bn_b'],[ov_pre,'bn_x']};
net.addLayer(layername, dagnn.BatchNorm('numChannels', convBlock1.size(4)), iv_name, ov_name, params);

%relu1
iv_name=ov_name;
layername=sprintf('relu_conv%d_1',d);ov_name=layername;
net.addLayer(layername, dagnn.ReLU(), iv_name, ov_name, {});

%dropout
iv_name=ov_name;
layername=sprintf('dropout_%d_1',d);ov_name=layername;
dropBlock=dagnn.DropOut('rate',0.3,'frozen',false);
net.addLayer(layername, dropBlock, iv_name, ov_name, {});

%conv2
iv_name=ov_name;
layername=sprintf('d_conv%d_2',d);ov_name=layername;
param=sprintf('d_conv%d_2_params',d);
convBlock2 = dagnn.Conv('size', [3,3,fD(d), fD(d)], 'hasBias', false, 'pad',[1,1,1,1]);
net.addLayer(layername, convBlock2, iv_name, ov_name, param);

%BN2
iv_name=ov_name;
ov_pre=layername;layername=[ov_pre,'_bn'];ov_name=layername;
params={[ov_pre,'bn_m'], [ov_pre,'bn_b'],[ov_pre,'bn_x']};
net.addLayer(layername, dagnn.BatchNorm('numChannels', convBlock2.size(4)), iv_name, ov_name, params);

%relu2
iv_name=ov_name;
layername=sprintf('relu_conv%d_2',d);ov_name=layername;
net.addLayer(layername, dagnn.ReLU(), iv_name, ov_name, {});

end

