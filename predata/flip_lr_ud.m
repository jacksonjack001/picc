pre='data/picc/mask/';
a=dir(pre);
for i=3:numel(a)
    str1=a(i).name;
    str=str1(1:end-4);
    I=imread([pre,str1]);
    I=fliplr(I);
    imwrite(I,['data/picc/mask/',str,'_lr','.png']);
end

fid=fopen('data/picc/ImageSets/dg_train.txt','w');
indeX=randperm(numel(a)-2)+2;
for i=indeX(1:end-24)
    str1=a(i).name;
    str=str1(1:end-4);
    fprintf(fid,str);
    fprintf(fid,'\n');
end
fclose(fid);

fid=fopen('data/picc/ImageSets/dg_val.txt','w');
for i=indeX(end-23:end)
    str1=a(i).name;
    str=str1(1:end-4);
    fprintf(fid,str);
    fprintf(fid,'\n');
end
fclose(fid);




pre='data/picc/CXR/';
a=dir('data/picc/CXR/');
for i=3:numel(a)
    str1=a(i).name;
    str=str1(1:end-4);
    I=imread([pre,str1]);
    I=fliplr(I);
    imwrite(I, ['data/picc/CXR/',str,'_lr','.jpg']);
end


