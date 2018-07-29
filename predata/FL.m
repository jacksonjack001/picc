x=[0:0.01:1];
f1=@(p)-(1-p).^2.*log(p);
y1=f1(x);


f2=@(p)-log(p);
y2=f2(x);

plot(x,y1,x,y2,x,y2-y1)
legend('fL','loss','diffenence')

[f1(0.2) f2(0.2)]

[f1(0.4) f2(0.4)]

[ind,m]=max(y2-y1);
x(m)