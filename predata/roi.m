pre='data/picc/ROI/';
a=dir('data/picc/ROI/');
for i=3:numel(a)
    str1=a(i).name;
    str=str1(1:end-3);
    I=imread([pre,str1]);
    I=rgb2gray(I);
    I=I(:,:);
    t=5;
    I(I<=t)=0;I(I>t)=255;
    
    BWI = imfill(I,'holes');
   imwrite(BWI,['data/picc/ROI0/',str,'png']);
%     copyfile(['picc/ROI/',str1],['picc/ROI1/',str2]); 
end
