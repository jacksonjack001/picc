Ki=2;Ko=3;Bn=2;
Xsize=3;Fsize=3;

X=rand(Xsize,Xsize,Ki,Bn);
F=rand(Fsize,Fsize,Ki,Ko);
B=zeros(Ko,1)+1;

% 前向过程
Y=vl_nnconv(X,F,B);

Y1=zeros(size(X,1)-2,size(X,2)-2,Ko,Bn);
for b=1:Bn
    for ko=1:Ko
        for ki=1:Ki
            % add by ki
            Y1(:,:,ko,b)=Y1(:,:,ko,b)+...
                convn(X(:,:,ki,b),rot90(F(:,:,ki,ko),2),'valid');
        end
        Y1(:,:,ko,b)=Y1(:,:,ko,b)+B(ko);
    end
end
Y(:)-Y1(:);

% 反向过程
dY=0.5*rand(size(Y));
[dX, dF, dB] = vl_nnconv(X, F, B, dY) ;

dB1=zeros(Ko,1);
dF1=zeros(size(F));
dX1=zeros(size(X));

for ko=1:Ko
    for b=1:Bn
        temp=dY(:,:,ko,b);
        dB1(ko)=dB1(ko)+sum(temp(:));
    end
end
dB,dB1

for ki=1:Ki
    for ko=1:Ko
        for b=1:Bn
            dF1(:,:,ki,ko)=dF1(:,:,ki,ko)+...
                convn(X(:,:,ki,b),rot90(dY(:,:,ko,b),2),'valid');
        end
    end
end
% dF(1:3),dF1(1:3)
dF(:)-dF1(:)


for ki=1:Ki
    for b=1:Bn
        for ko=1:Ko
            dX1(:,:,ki,b)=dX1(:,:,ki,b)+...
                convn(F(:,:,ki,ko),rot90(dY(:,:,ko,b),2),'valid');
        end
    end
end
dX(:)-dX1(:)