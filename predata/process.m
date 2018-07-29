mkdir data/picc/mask0
pre='data/picc/mask/';
a=dir('data/picc/mask/');
for i=3:numel(a)
    str1=a(i).name;
    str=str1(1:end-3);
    I=imread([pre,str1]);
    I=rgb2gray(I);
    I=I(:,:);
    t=5;
    I(I<=t)=0;I(I>t)=255;
    imwrite(I,['data/picc/mask0/',str,'png']);
end



% imdb = vocSetup1();

mkdir data/picc/ROI0
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
end


















