function net=denseNet1()

opts.networkType = 'dagnn' ;
opts.depth=40; 
opts.Nclass=10; 
opts.num_blocks=3;
opts.dropout=0;
opts.growth_rate=12;
% k0=16;

net= dagnn.DagNN; %network
n = (opts.depth - 1) / opts.num_blocks; %stacks of residual units
incoming = 'input';
convBlock1 = dagnn.Conv('size', [3,3,3,opts.growth_rate], 'pad', [1,1,1,1],'stride', [1,1], 'hasBias', false);
net.addLayer('pre_conv',convBlock1,incoming,{'pre_conv'},{'conv_pre_params'});

incoming='pre_conv';
for b=1:opts.num_blocks
    [net,ov_name]=dense_block(net,incoming,sprintf('block%d',b),n,opts.growth_rate,opts.dropout);
    if b<opts.num_blocks
        [net,ov_name]=transition(net,ov_name,sprintf('block%d_trs',b),opts.growth_rate,opts.dropout);
    end
    incoming=ov_name;
end

params1={'post_bn_m', 'post_bn_b','post_bn_x'};

% n_th_channels=k0+n*opts.growth_rate;
% f1=net.getVarIndex(ov_name);fv1=net.vars(f1).value;channels1=size(fv1,3);
channels1=opts.growth_rate;
net.addLayer('post_bn',dagnn.BatchNorm('numChannels', channels1),{ov_name}, {'post_bn'}, params1);
net.addLayer('post_relu',dagnn.ReLU(),{'post_bn'},{'post_relu'},{});

blockPool = dagnn.Pooling('method', 'avg', 'poolSize', [8 8], 'stride', 1, 'pad', [0,0,0,0]);
net.addLayer('post_pool', blockPool, {'post_relu'}, {'post_pool'}, {}) ;

% f2=net.getVarIndex('post_pool');fv2=net.vars(f2).value;channels2=size(fv2,3);
channels2=opts.growth_rate;
convBlock1 = dagnn.Conv('size', [1,1,channels2,opts.Nclass], 'pad', [0,0,0,0],'stride', [1,1], 'hasBias', false);
net.addLayer('prediction',convBlock1,{'post_pool'},{'prediction'},{'Dense_params'});

%Loss layer
net.addLayer('objective', dagnn.Loss('loss', 'softmaxlog'), {'prediction','label'}, 'objective') ;

%Error layer 
net.addLayer('error', dagnn.Loss('loss', 'classerror'), {'prediction','label'}, 'error') ;

net.initParams() ;
    function   [net,ov_name]=dense_block(net,iv_name,db_ov_pre,n,growth_rate,dropout)
        %n表示每一denseBlock块儿有多少个卷积层
        for i=1:n
%             i
            [net,ov_name]=bn_relu_conv(net,iv_name,[db_ov_pre,sprintf('_ly%02d',i)],growth_rate,3,dropout,1);
            net.addLayer([db_ov_pre,sprintf('_ly%02d_join',i)], dagnn.Concat('dim',3), {iv_name,ov_name}, ...
                {[db_ov_pre,sprintf('_ly%02d_join',i)]});
            ov_name=[db_ov_pre,sprintf('_ly%02d_join',i)];
            iv_name=ov_name;
        end
    end

    function [net,ov_name]=bn_relu_conv(net,iv_name,ov_name_pre,channels,filter_size,dropout,pad_size)
        params={[ov_name_pre,'bn_m'], [ov_name_pre,'bn_b'],[ov_name_pre,'bn_x']};
        net.addLayer([ov_name_pre,'_bn'],dagnn.BatchNorm('numChannels', channels),{iv_name}, {[ov_name_pre,'_bn']},params);
        net.addLayer([ov_name_pre,'_relu'],dagnn.ReLU(),{[ov_name_pre,'_bn']}, {[ov_name_pre,'_relu']},{});
        convBlock = dagnn.Conv('size', [filter_size,filter_size,channels,channels], 'pad', [pad_size,pad_size,pad_size,pad_size],'stride', [1,1], 'hasBias', false);
        net.addLayer([ov_name_pre,'_conv'],convBlock,{[ov_name_pre,'_relu']},{[ov_name_pre,'_conv']},{[ov_name_pre,'_conv_params']});
        ov_name=[ov_name_pre,'_conv'];
    end

    function [net,ov_name]=transition(net,iv_name,ov_name_pre,channels,dropout)
%         net.initParams() ;f=net.getVarIndex(iv_name);fv=net.vars(f).value;channels=size(fv,3);
        [net,ov_name]=bn_relu_conv(net,iv_name,ov_name_pre,channels,1,dropout,0);
        iv_name=ov_name; 
        poolBlock = dagnn.Pooling('method', 'avg', 'poolSize', [2 2], 'stride', [2,2]);
        net.addLayer([ov_name_pre,'_pool'], poolBlock, {iv_name},{[ov_name_pre,'_pool']},{});
        ov_name=[ov_name_pre,'_pool'];
    end

end
