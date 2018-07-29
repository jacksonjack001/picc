%YH = floor((H + (PADTOP+PADBOTTOM) - FH)/STRIDEY) + 1,
%YW = floor((W + (PADLEFT+PADRIGHT) - FW)/STRIDEX) + 1.
%   B is a SINGLE array with 1 x 1 x K elements (B can in fact
%   be of any shape provided that it has K elements).
X=rand(30,30,3,1);
F=rand(3,3,3,10);
B=zeros(1,10);%B=zeros(10,1);
cX=vl_nnconv(X,F,B);

%   Alternatively, FC can
%   *divide* the C; in this case, filters are assumed to form G=C/FC
%   *groups* of equal size (where G must divide K). Each group of
%   filters works on a consecutive subset of feature channels of the
%   input array X.
X=rand(30,30,30,1);
F=rand(3,3,3,20);
G=30/3;G|20;
B=zeros(1,20);%B=zeros(10,1);
cX=vl_nnconv(X,F,B);

%YH = UPH (XH - 1) + FH - CROPTOP - CROPBOTTOM,
%YW = UPW (XW - 1) + FW - CROPLEFT - CROPRIGHT.
X=rand(12,16,21,1);
F=rand(64,64,1,21);
vl_nnconvt(X,F,[],'crop',[16,16,16,16],'Upsample',32)

%等价性
U=vl_nnconv(V,F,[]);
U(:)=M*V(:)
%那么
Y(:)=M'*X(:)


X=rand(3,3,3,2);
G=ones(3,1);
B=zeros(3,1);
Y = vl_nnbnorm(X,G,B)