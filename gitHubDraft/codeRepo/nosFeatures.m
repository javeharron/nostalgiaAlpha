x=[64 65 448 449 450 63  64  65  99 100 485 63 64 65 63 64 65 63 64 65 448 449 450 63 64 65 63  64  65 240 274 275 63 64 65 63  64  65 274 275 450 63 64 65 30  64  65 275 449 450];
x0=unique(x);
%figure;
%histogram(x)

M = csvread('NOS001_Data.csv');

%py: features 30 (chan 1), 63-65 (chan2)
%matlab: 30>32, 63-65>65-67

p1=M(:,32);

p2=M(:,65:67);
p=.92;
N=12;

br=log2(N)+(p*log2(p))+(1-p)*log2((1-p)/(N-1));
itr=br*60;