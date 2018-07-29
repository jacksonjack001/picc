label=single(imread('data/picc/mask/Xu_Kuanlang.png'));
label=label/255;label=mod(label+1,256);
sz=[1024,512,256,128,64];
loss=[];F={};
for i=1:5
    L{i}=imresize(label,[sz(i),sz(i)]);
    L{i}=round(L{i});
end
mkdir('unet_fgm');

for k=1:10
    for epoch=(k-1)*10+[1:10]
        epoch
        for d=1:6
            A=[];
            directory=['./unet',num2str(epoch*5),'/', num2str(d),'.mat'];
            load(directory);
            A=tt;
            F{epoch,d}=fgm(A,2);
            subplot(1,2,1);imshow(F{epoch,d}(:,:,1),[]);
            subplot(1,2,2);imshow(F{epoch,d}(:,:,2),[]);
            
            saveas(gca,['unet_fgm/',num2str(epoch),'_',num2str(d),'.png']);
            %         nsz=size(F{epoch,d},1);
            %         ind=log2(1024/nsz)+1;
            %         mass = sum(sum(L{ind} > 0,2),1) + 1 ;
            %         loss(epoch,d)=vl_nnloss(F{epoch,d},L{ind}, [], 'loss','softmaxlog', ...
            %             'instanceWeights', 1./mass, 'classWeights', [1]);
        end
    end
    
end

pre='unet_fgm/';
for k=1:10%
    T=[];
    for iter=(k-1)*10+[1:10]
        T1=[];
        for i=1:6
            name=[pre,num2str(iter),'_',num2str(i),'.png'];
            I=imread(name);
            T1=[T1 I];
        end
        T=[T;T1];
    end
    imwrite(T,[pre,'aa',num2str(k),'.png']);
end






plot(loss(:,1:4))
ld={'t4r','t3r','t2r','t1r','pred'};
legend(ld(1:4))


% i=1;
% for d=1:10
%     subplot(10,2,i)
%     imshow(F{epoch,d}(:,:,1),[])
%     subplot(10,2,i+1)
%     imshow(F{epoch,d}(:,:,2),[])
%     i=i+2;
% end

% plot(loss)
% legend('b5','t4b','t4r','t4b','t3r','t3b','t2r','t2b','t1r','t1b','pred')
%
% plot(loss(:,[1:2:9]))
% ld={'b5','t4b','t4r','t4b','t3r','t3b','t2r','t2b','t1r','t1b','pred'};
% legend(ld(1:2:9))