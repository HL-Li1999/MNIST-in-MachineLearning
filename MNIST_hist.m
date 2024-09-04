%by Lihuanlin 2022/11
%程序功能：读取MNIST训练集样本，画出各样本点与对应重心的距离的直方图

%% 设置参数
    %直方图中bin的个数
    numOfBins = 10;
%% 读取训练集中所有样本
    train_Path = 'G:\研究生\实验资料\机器学习\数据集\手写数字MNIST\MNIST_bmp\train_img\';   
    train_File = dir(fullfile(train_Path,'*.bmp'));  
    train_FileNames = {train_File.name}';   
%% 预处理：各分类样本数
    N=zeros(10,1);
    for i=1:length(train_FileNames)
        num=str2double(train_FileNames{i}(1));
        N(num+1)=N(num+1)+1;
    end
    M=[0;N];    
%% 保存训练集所有样本点
    Xn=zeros(784,60000);
    for i=1:length(train_FileNames)
        Img=imread(strcat(train_Path,train_FileNames{i}));
        x=im2double(Img(:)); 
        Xn(:,i)=x;
    end    
%% 计算训练集重心位置
    Average=zeros(784,10);
    for i=1:10
        COL1=sum(M(1:i))+1;
        COL2=sum(M(1:i+1));
        Average(:,i)=sum(Xn(:,COL1:COL2),2)/N(i);
    end
%     for i=1:10
%         AveragePic=reshape(Average(:,i),28,28);
%         mat2gray(AveragePic);
%         figure(i);
%         imshow(AveragePic);
%         title(num2str(i));
%     end
%% 计算各样本到对应中心的距离
    dist=zeros(60000,1);
    for i=1:10
        for j=1:N(i)
            picnum=sum(M(1:i))+j;
            dist(picnum)=sqrt(sum((Xn(:,picnum)-Average(:,i)).^2));
        end
    end    
%% 显示直方图
     hold off
    for i=10:10
        NUM1=sum(M(1:i))+1;
        NUM2=sum(M(1:i+1));
        [histFreq, histXout] = hist(dist(NUM1:NUM2), numOfBins);
        binWidth = histXout(2)-histXout(1);
        figure(i+10);
        bar(histXout, histFreq);       
        xlabel('distance');
        ylabel('K(distance)');
%         hold on

    end
%% 在直方图中显示高斯函数
%     for i=1:10
%         figure(i+10);
%         x=4:14;
%         y=normpdf(x,7,1);
%         plot(x,y);
%     end    