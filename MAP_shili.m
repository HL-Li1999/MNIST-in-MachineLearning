%�����ܣ�����MAP����϶�Ԫ�����Ժ���
%% ���ò���
miu=1;
%% �������ݳ�ʼ��
x1=-5:0.1:5;
x2=-5:0.1:5;
%����������x1��x2ת��Ϊ������
[x1,x2]=meshgrid(x1,x2); 
t=4*x1.^2+5*x2.^2+6;
figure(1);
mesh(x1,x2,t);
title('��������ͼ');
%% ������t(������)
t=t+randn(size(t));
t=t(:);
%% �����
fx0=ones(size(x1));
fx1=x1.^2;
fx2=x2.^2;
fhi=[fx0(:),fx1(:),fx2(:)];
%% ������Wmap
Wmap=(miu*eye(size(fhi'*fhi))+fhi'*fhi)\fhi'*t;
%% ������y(x;w)
y=fhi*Wmap;
%% ������ƽ��������Ϊ���
R=sum((y-t).^2)/length(y);
%% �����ͼ
y=reshape(y,101,101);
figure(2);
mesh(x1,x2,y);
title('�������ͼ');