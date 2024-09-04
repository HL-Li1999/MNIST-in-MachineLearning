%by Lihuanlin 2022/11
%程序功能：实验测试Metropolis-Hastings算法的采样效果
%原理：用二维正态分布（提议分布）模拟另外一个二维正态分布（目标分布）

%% 预生成已知二维样本点
    %实际分布为二维高斯分布N(u1,∑1)，共有1000个样本点
    Average1=[0;0];
    Variance1=4*eye(2,2);
    PX=mvnrnd(Average1,Variance1,10000);
    %% 显示二维样本点位置
    figure(1);
    plot(PX(:,1),PX(:,2),'+');
    title('实际样本点');   
    %% 显示两个一维直方图
    %x方向
    numOfBins = 100;
    [histFreq, histXout] = hist(PX(:,1), numOfBins);
    binWidth = histXout(2)-histXout(1);
    figure(2);
    bar(histXout, histFreq/binWidth/sum(histFreq));       
    xlabel('xdistance');
    ylabel('N(xdistance)');
    hold on;
    %y方向
    [histFreq, histXout] = hist(PX(:,2), numOfBins);
    binWidth = histXout(2)-histXout(1);
    figure(3);
    bar(histXout, histFreq/binWidth/sum(histFreq));       
    xlabel('ydistance');
    ylabel('N(ydistance)');
    hold on
    %% 在直方图上显示高斯函数
    %x方向
    figure(2);
    dx=0.5;
    xx=-8:dx:8;
    yy=normpdf(xx,0,2);
    plot(xx,yy);
    %y方向
    figure(3);
    yy=normpdf(xx,0,2);
    plot(xx,yy);    

%% 由已知样本点估计目标分布p(x)
    %假设目标分布为二维高斯分布N(u2,∑2)，u2,∑2由已有样本点估计
    Average2x=sum(PX(:,1))/1000;
    Average2y=sum(PX(:,2))/1000;
    Average2=[Average2x;Average2y];
    Variance2=zeros(2,2);
    for i=1:1000
        Variance2=Variance2+(PX(i,:)'-Average2)*(PX(i,:)'-Average2)'/1000;
    end
           
%% 
    %提议分布q(x)取二维高斯分布N(x,∑3)，共抽k个点
    k=1000;
    Variance3=1*eye(2,2);
    %储存抽样点的矩阵
    X=zeros(2,k);
    for i=1:k-1
        x=X(:,i);
        xs=mvnrnd(x',Variance3)';
        %目标分布p(x)为N(u2,∑2)
        A=min(1,mvnpdf(xs',Average2',Variance2)/mvnpdf(x',Average2',Variance2));
        u=rand;
        if A>u
            X(:,i+1)=xs;
        else
            X(:,i+1)=x;
        end
    end
%% 显示抽样矩阵两个一维直方图
    %x方向 
    figure(4);
    nb=histc(X(1,1:k-1),xx); 
    bar(xx+dx/2,nb/(k-1)/dx); 
    title('x');
    hold on;
    %y方向
    figure(5);
    nb=histc(X(2,1:k-1),xx); 
    bar(xx+dx/2,nb/(k-1)/dx); 
    title('y');
    hold on
    %% 在直方图上显示高斯函数
    %x方向
    figure(4);
    x=-8:8;
    y=normpdf(x,0,2);
    plot(x,y);
    %y方向
    figure(5);
    y=normpdf(x,0,2);
    plot(x,y); 
    hold off;