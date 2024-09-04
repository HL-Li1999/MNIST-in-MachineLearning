%程序功能：利用MAP法拟合二元非线性函数
%% 设置参数
miu=1;
%% 样本数据初始化
x1=-5:0.1:5;
x2=-5:0.1:5;
%将坐标向量x1、x2转换为格点矩阵
[x1,x2]=meshgrid(x1,x2); 
t=4*x1.^2+5*x2.^2+6;
figure(1);
mesh(x1,x2,t);
title('理想数据图');
%% 求向量t(含噪声)
t=t+randn(size(t));
t=t(:);
%% 求矩阵Ф
fx0=ones(size(x1));
fx1=x1.^2;
fx2=x2.^2;
fhi=[fx0(:),fx1(:),fx2(:)];
%% 求向量Wmap
Wmap=(miu*eye(size(fhi'*fhi))+fhi'*fhi)\fhi'*t;
%% 求向量y(x;w)
y=fhi*Wmap;
%% 求误差：以平均方差作为误差
R=sum((y-t).^2)/length(y);
%% 画拟合图
y=reshape(y,101,101);
figure(2);
mesh(x1,x2,y);
title('拟合数据图');