modelPath1 = 'net-epoch-100-ml_1p1_test40_40_20.mat';%navie
modelPath2 = 'net-epoch-100-illaverage-test40-40-20.mat';%multi-Loss

stats1=load(modelPath1,'stats') ;train1=stats1.stats.train;a1=stats1.stats.val;
stats2=load(modelPath2,'stats') ;train2=stats2.stats.train;a2=stats2.stats.val;

ep=100;accind=6;
values=zeros(ep,4);
for i=1:ep
    values(i,1)=gather(train1(i).accuracy(3));
    values(i,2)=gather(train2(i).accuracy(3));
    values(i,3)=gather(a1(i).accuracy(3));
    values(i,4)=gather(a2(i).accuracy(3));
end
x=1:ep;
plot(x,values(:,1),'r-',...
    x,values(:,2),'g-',...
    x,values(:,3),'r-.',...
    x,values(:,4),'g-.');
legend('navie-mIou-trn','mL-mIou-trn','navie-mIou-val','mL-mIou-val')
saveas(gca,'mIou.png');



values=zeros(ep,4);
for i=1:ep
    values(i,1)=train1(i).objective(1);
    values(i,2)=train2(i).objective(1);
    values(i,3)=a1(i).objective(1);
    values(i,4)=a2(i).objective(1);
end
x=1:ep;
plot(x,values(:,1),'r-',...
    x,values(:,2),'g-',...
    x,values(:,3),'r-.',...
    x,values(:,4),'g-.');
legend('navieLoss-trn','mLoss-trn','navieLoss-val','mLoss-val')
saveas(gca,'loss.png');
