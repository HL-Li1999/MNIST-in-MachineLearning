%by Lihuanlin 2022/11
%程序功能：对MNIST测试集做K均值聚类

%% 读取测试集样本
    Path = 'G:\研究生\实验资料\机器学习\数据集\手写数字MNIST\MNIST_bmp\test_img\';   
    File = dir(fullfile(Path,'*.bmp'));  
    FileNames = {File.name}';
    Xn=zeros(784,length(FileNames));
    N=zeros(10,1);
    for i=1:length(FileNames)
        Img=imread(strcat(Path,FileNames{i}));
        x=im2double(Img(:));
        Xn(:,i)=x;
        num=str2double(FileNames{i}(1))+1;
        N(num)= N(num)+1;
    end 
%% 初始化参数
    K=10;
    itnum=100;
    u=rand(784,K); 
    dist_u=zeros(itnum,1);
    dist_min=zeros(itnum,1);
    index=zeros(10,itnum);    
%% 求K=10时的理想聚类中心
    M=[0;N];
    Average=zeros(784,10);
    for i=1:10
        NUM1=sum(M(1:i))+1;
        NUM2=sum(M(1:i+1));
        Average(:,i)=sum(Xn(:,NUM1:NUM2),2)/N(i);
    end
    %% 迭代
    for iterate=1:itnum    
        %% 固定u，确定每个样本所属类别z
        u0=u;
        dist=zeros(10,length(FileNames));
        for i=1:K
            uu=repmat(u(:,i),1,length(FileNames));
            dist(i,:)=sum((Xn-uu).^2);
        end
        z=zeros(10,length(FileNames));
        for i=1:length(FileNames)
            [value,pos]=min(dist(:,i));
            z(pos,i)=1;
        end  
        %% 固定z，求u
        for i=1:K 
            zi=repmat(z(i,:),784,1);
            Xk=zi.*Xn;
            nk=sum(z(i,:));
            if nk==0
                nk=1;
            end
            u(:,i)=sum(Xk,2)/nk;
        end
        %% 聚类中心的每次总移动距离
        dist_u(iterate)=sum(sqrt(sum((u-u0).^2)));       
        %% 聚类中心与理想聚类中心的最近距离
        Dist_min=zeros(10,1);
        for a=1:10
            %搜索各个实际聚类中心最近的理想聚类中心(可能重复)
            [idx,id]=knnsearch(Average',u(:,a)','k',1);
            Dist_min(a)=sum(sqrt(sum((Average(:,idx)-u(:,a)).^2)));
            %用以查看与实际聚类中心最近的理想聚类中心
            index(a,iterate)=idx;
        end
        dist_min(iterate)=sum(Dist_min);                       
    end  
    %% 画图
    i=1:itnum;
    figure(1)
    plot(i',dist_u);
    title('聚类中心的每次总移动距离');
    figure(2)
    plot(i',dist_min);
    title('聚类中心与理想聚类中心的最近距离');