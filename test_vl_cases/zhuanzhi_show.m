X=net.vars(39).value;   %12    16    21
X1=net.params(33).value;%64    64     1    21

% 池化前的卷积层
iter=[8, 15, 22, 29, 39];
n=iter(3);
in=net.layers(n).inputs;
on=net.layers(n).outputs;
wn=net.layers(n).params{1};
bn=net.layers(n).params{2};

iv=net.vars(net.getVarIndex(in)).value;
ov=net.vars(net.getVarIndex(on)).value;
wp=net.params(net.getParamIndex(wn)).value;
bp=net.params(net.getParamIndex(bn)).value;
% subplot(1,2,1);imshow(ov(:,:,1),[]);
% Y2 = vl_nnconv(iv, wp,bp, 'Pad',[1,1,1,1]);
% subplot(1,2,2);imshow(Y2(:,:,1),[]);
% Y2./ov

size(ov),size(wp),size(bp),size(Y2)
Y2 = vl_nnconvt(ov, wp, bp, 'crop', [1 0 1 0],'Upsample',2);
imshow(Y2(:,:,1),[])


ov=rand(48,64,512);
wp=ones(1,1,512,512);
bp=zeros(512,1);
Y2 = vl_nnconvt(ov, wp,bp,'crop', [0 0 0 0],'Upsample',2);
Y2(1:3,1:3,1), ov(1:3,1:3,1)
