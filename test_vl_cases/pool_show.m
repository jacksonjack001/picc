A=randi(10,10,10);
[X,Y]=meshgrid(1:5);
grid=[];
grid(1,:,:)=X;grid(2,:,:)=Y;
y=vl_nnbilinearsampler(A,grid);


Y = vl_nnpool(A, [2, 2],'stride',[2,2]);



2
0 0
23 59
1
23 59
