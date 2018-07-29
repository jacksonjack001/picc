mkdir data/picc/ROI1

pre='data/picc/ROI/';
loc='data/picc/ROI1/';
a=dir('data/picc/ROI/');
for i=3:numel(a)
    str1=a(i).name;
    str=strrep(str1,'_ROI','')
    copyfile([pre,str1],[loc,str]);    
end



strrep('Lai_Baoxing_RO.png','_ROI','')


mkdir data/picc/mask1

pre='data/picc/mask/';
loc='data/picc/mask1/';
a=dir('data/picc/mask/');
for i=3:numel(a)
    str1=a(i).name;
    str=strrep(str1,'_Remark','')
    copyfile([pre,str1],[loc,str]);    
end





mkdir data/picc/CXR1

pre='data/picc/CXR/';
loc='data/picc/CXR1/';
a=dir('data/picc/CXR/');
for i=3:numel(a)
    str1=a(i).name;
    str=str1(1:end-10);  
    copyfile([pre,str1],[loc,str,'.jpg']);    
end



imdb = vocSetup1();



















