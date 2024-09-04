%程序功能：采用基本Metropolis对MNIST样本集进行抽样

%% 导入的训练集所有样本
    Path ='G:\研究生\实验资料\机器学习\数据集\手写数字MNIST\MNIST_bmp\train_img\';   
    File = dir(fullfile(Path,'*.bmp'));  
    FileNames = {File.name}'; 
    
%% 预处理:获取各分类图片数
    N=zeros(10,1);
    for i=1:length(FileNames)
        num=str2double(FileNames{i}(1));
        N(num+1)=N(num+1)+1;
    end
    M=[0;N];
    
%% 求目标分布参数，目标分布设为多维高斯分布N(Average,Variance)，建议分布为N(u,Sigma)
    % 求u
    x=zeros(784,1);
    Average=zeros(784,10);
    for i=1:10
        for j=1:N(i)
            pic_num=sum(M(1:i))+j;           
            Img=imread(strcat(Path,FileNames{pic_num}));
            x=im2double(Img(:));           
            Average(:,i)=Average(:,i)+x/N(i);             
        end
    end
    %求∑
    Variance=zeros(784,784);
    for i=1:10
        for j=1:N(i)
            pic_num=sum(M(1:i))+j;
            Img=imread(strcat(Path,FileNames{pic_num}));
            x=im2double(Img(:));    
            Variance=Variance+(x-Average(:,i))*(x-Average(:,i))'/sum(N);
        end  
    end
%     Variance=Variance+0.01*eye(784,784);
%% Metropolis-Hastings
    num=100;
    Sigma=eye(784,784);
    H=Variance+10*max(diag(Variance))*eye(784,784);
    for i=1:10
        X=zeros(784,num);
        %% 保存一个分类的抽样
        for j=1:num-1
            x=X(:,j);
            px=mvnpdf(x',Average(:,i)',H);
            xs=mvnrnd(x,Sigma)';
            pxs=mvnpdf(xs',Average(:,i)',H);
            A=min(1,pxs/px);
            u=rand;
            if A>u
                X(:,j+1)=xs;
            else
                X(:,j+1)=x;
            end 
        end
        %将抽样得到的样本以txt形式输出
        [m, n] = size(X);
        SavePath='C:\Users\llll\Desktop\1\';
        txtname=[num2str(i-1),'.txt'];
        fid=fopen(strcat(SavePath, txtname), 'wt');
        for j= 1 : m
            fprintf(fid, '%g\t', X(j,:));
            fprintf(fid, '\n');
        end
    end

%% 显示k方向高斯函数
    k=491;
    figure(1);
    dx=0.5;
    xx=-30:dx:30;
    c1=5;
    c2=5;
    yy=normpdf(xx,Average(k,10)+c1,H(k,k)+c2);
    plot(xx,yy);
    hold on;    
    
%   显示抽样矩阵的k方向直方图
    %xk方向
    figure(1);
    nb=histc(X(k,1:num),xx); 
    bar(xx+dx/2,nb/num/dx); 
    hold off;
