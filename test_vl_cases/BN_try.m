Ki=5;Ko=5;Bn=3;Xsize=2;

X=rand(Xsize,Xsize,Ki,Bn);
G=ones(Ki,1);
B=zeros(Ki,1);
Y = vl_nnbnorm(X,G,B);


mu=f3m(X);
tX=X;sX=zeros(size(tX));
for k=1:Ki
    sX(:,:,k,:)=(tX(:,:,k,:)-mu(:,:,k,:)).^2;
end
sigma2=f3m(sX);

EPSILON=1e-4;
sigma_2=EPSILON+sigma2;

xB=zeros(size(X));
for k=1:Ki
    xB(:,:,k,:)=(X(:,:,k,:)-mu(:,:,k,:))./sqrt(sigma_2(k));
end

Y1=[];
for k=1:Ki
    Y1(:,:,k,:)=G(k)*xB(:,:,k,:)+B(k);
end
Y(:)-Y1(:)

dY=rand(size(X));%[1,2;3,4];%
[dX, dG, dB] = vl_nnbnorm(X, G, B, dY);

% in BN  Ki=Ko
dB1=zeros(size(B));
for ko=1:Ki
    for b=1:Bn
        temp=dY(:,:,ko,b);
        dB1(ko)=dB1(ko)+sum(temp(:));
    end
end

dG1=zeros(size(G));
for ko=1:Ki
    for b=1:Bn
        temp=dY(:,:,ko,b).*xB(:,:,ko,b);
        dG1(ko)=dG1(ko)+sum(temp(:));
    end
end
dG-dG1


m=Bn;
pre=1./(sqrt(squeeze(sigma_2)));
dX1=zeros(size(X));
for ko=1:Ki
    for b=1:Bn
    dy=dY(:,:,ko,b);
    tm1=dY(:,:,ko,:);
    m1=mean(mean(mean(tm1)));
    tm2=dY(:,:,ko,:).*Y(:,:,ko,:);
    m2=mean(mean(mean(tm2)));
    M2=m2*Y(:,:,ko,b);
    dX1(:,:,ko,b)=pre(ko)*(dy-m1-M2);
    end
end
dX(:)-dX1(:)
