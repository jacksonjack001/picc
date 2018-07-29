X=rand(10);
[Y,MASK] = vl_nndropout(X)
sum(sum(MASK>0))