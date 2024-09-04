%by Lihuanlin 2022/11
%�����ܣ�ʵ�����Metropolis-Hastings�㷨�Ĳ���Ч��
%ԭ���ö�ά��̬�ֲ�������ֲ���ģ������һ����ά��̬�ֲ���Ŀ��ֲ���

%% Ԥ������֪��ά������
    %ʵ�ʷֲ�Ϊ��ά��˹�ֲ�N(u1,��1)������1000��������
    Average1=[0;0];
    Variance1=4*eye(2,2);
    PX=mvnrnd(Average1,Variance1,10000);
    %% ��ʾ��ά������λ��
    figure(1);
    plot(PX(:,1),PX(:,2),'+');
    title('ʵ��������');   
    %% ��ʾ����һάֱ��ͼ
    %x����
    numOfBins = 100;
    [histFreq, histXout] = hist(PX(:,1), numOfBins);
    binWidth = histXout(2)-histXout(1);
    figure(2);
    bar(histXout, histFreq/binWidth/sum(histFreq));       
    xlabel('xdistance');
    ylabel('N(xdistance)');
    hold on;
    %y����
    [histFreq, histXout] = hist(PX(:,2), numOfBins);
    binWidth = histXout(2)-histXout(1);
    figure(3);
    bar(histXout, histFreq/binWidth/sum(histFreq));       
    xlabel('ydistance');
    ylabel('N(ydistance)');
    hold on
    %% ��ֱ��ͼ����ʾ��˹����
    %x����
    figure(2);
    dx=0.5;
    xx=-8:dx:8;
    yy=normpdf(xx,0,2);
    plot(xx,yy);
    %y����
    figure(3);
    yy=normpdf(xx,0,2);
    plot(xx,yy);    

%% ����֪���������Ŀ��ֲ�p(x)
    %����Ŀ��ֲ�Ϊ��ά��˹�ֲ�N(u2,��2)��u2,��2���������������
    Average2x=sum(PX(:,1))/1000;
    Average2y=sum(PX(:,2))/1000;
    Average2=[Average2x;Average2y];
    Variance2=zeros(2,2);
    for i=1:1000
        Variance2=Variance2+(PX(i,:)'-Average2)*(PX(i,:)'-Average2)'/1000;
    end
           
%% 
    %����ֲ�q(x)ȡ��ά��˹�ֲ�N(x,��3)������k����
    k=1000;
    Variance3=1*eye(2,2);
    %���������ľ���
    X=zeros(2,k);
    for i=1:k-1
        x=X(:,i);
        xs=mvnrnd(x',Variance3)';
        %Ŀ��ֲ�p(x)ΪN(u2,��2)
        A=min(1,mvnpdf(xs',Average2',Variance2)/mvnpdf(x',Average2',Variance2));
        u=rand;
        if A>u
            X(:,i+1)=xs;
        else
            X(:,i+1)=x;
        end
    end
%% ��ʾ������������һάֱ��ͼ
    %x���� 
    figure(4);
    nb=histc(X(1,1:k-1),xx); 
    bar(xx+dx/2,nb/(k-1)/dx); 
    title('x');
    hold on;
    %y����
    figure(5);
    nb=histc(X(2,1:k-1),xx); 
    bar(xx+dx/2,nb/(k-1)/dx); 
    title('y');
    hold on
    %% ��ֱ��ͼ����ʾ��˹����
    %x����
    figure(4);
    x=-8:8;
    y=normpdf(x,0,2);
    plot(x,y);
    %y����
    figure(5);
    y=normpdf(x,0,2);
    plot(x,y); 
    hold off;