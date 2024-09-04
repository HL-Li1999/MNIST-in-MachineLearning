%by Lihuanlin 2022/11
%程序用途：用核密度估计建立分类生成模型，样本用MNIST数据集

%% 设定参数
    %取平滑参数h的值
    h=1;
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
%% 读取测试集中所有样本
    %抽样过，每一个txt文件为所有抽样的同类样本点的矩阵
    test_Path = 'G:\研究生\实验资料\机器学习\数据集\手写数字MNIST\MNIST_bmp\test_img\';   
    test_File = dir(fullfile(test_Path,'*.bmp'));  
    test_FileNames = {test_File.name}';
    
%% 核密度估计
    %% 保存训练集所有样本点
    Xn=zeros(784,60000);
    for i=1:length(train_FileNames)
        Img=imread(strcat(train_Path,train_FileNames{i}));
        x=im2double(Img(:)); 
        Xn(:,i)=x;
    end
    %% 测试集输入一张图片：待观察目标
    correct=0;
    M=[0;N];
    for i=1:length(test_FileNames)
        Img=imread(strcat(test_Path,test_FileNames{i}));
        x=im2double(Img(:)); 
        %采用核密度估计求p(x|Ck)
        px=zeros(10,1);
        xx=repmat(x,1,sum(N));
        A=xx-Xn;
        B=sum(A.^2,1);
        C=-B/(2*h^2);
        for j=1:10
            COL1=sum(M(1:j))+1;
            COL2=sum(M(1:j+1));
            px(j)=sum(exp(C(:,COL1:COL2))/N(j));
        end
        %% 求P(Ck|x)    
        pCk=px.*N;
        %判断p(Ck|x_test)中最大值即为分类结果
        [max_value,max_pos]=max(pCk);
        if max_pos-1==str2double(test_FileNames{i}(1))
            correct=correct+1;
        end
    end
    rate=correct/length(test_FileNames);
            