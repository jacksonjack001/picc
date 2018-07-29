pre='data/picc/mask/';
a=dir('data/picc/val/ROI/');
for i=3:numel(a)
    str1=a(i).name;
    movefile(['data/picc/mask/',str1],['data/picc/val/mask/',str1]); 
end


m=dir(pre)
pre='data/picc/CXR/';
a=dir('data/picc/val/ROI/');
for i=3:numel(a)
    str1=a(i).name;
    str2=str1(1:end-4);
    movefile(['data/picc/CXR/',str2,'.jpg'],['data/picc/val/CXR/',str2,'.jpg']); 
end
