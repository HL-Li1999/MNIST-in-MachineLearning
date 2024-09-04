%by Lihuanlin 2022/11
%程序用途：用K最近邻建立分类生成模型，样本用MNIST数据

%% 设定参数
    %获取观测点最邻近的k个样本点
    k=3;
%% 读取训练集中所有图片
    train_Path = 'G:\研究生\实验资料\机器学习\数据集\手写数字MNIST\MNIST_bmp\train_img\';   
    train_File = dir(fullfile(train_Path,'*.bmp'));  
    train_FileNames = {train_File.name}';    
%% 预处理：各分类图片数
    N=zeros(10,1);
    for i=1:length(train_FileNames)
        num=str2double(train_FileNames{i}(1));
        N(num+1)=N(num+1)+1;
    end
    M=[0;N];
%% 读取测试集中所有样本
    test_Path = 'G:\研究生\实验资料\机器学习\数据集\手写数字MNIST\MNIST_bmp\test_img\';   
    test_File = dir(fullfile(test_Path,'*.bmp'));  
    test_FileNames = {test_File.name}';    
%% K最近邻
    %% 训练集输入所有图片：样本点
    Xn=zeros(784,60000);
    for i=1:length(train_FileNames)
        Img=imread(strcat(train_Path,train_FileNames{i}));
        x=im2double(Img(:)); 
        Xn(:,i)=x;
    end
    %% 测试集输入一张图片：观测点
    correct=0;
    for i=1:length(test_FileNames)
        Img=imread(strcat(train_Path,test_FileNames{i}));
        x_test=im2double(Img(:));
        [idx,id]= knnsearch(Xn',x_test','k',k);
        %作K最近邻分类判断
        resultNum=zeros(10,1);
        for i_idx=1:k
            xtrain_class=str2double(train_FileNames{idx(i_idx)}(1))+1;
            resultNum(xtrain_class)=resultNum(xtrain_class)+1;
        end
        [max_value,max_pos]=max(resultNum);
        if max_pos-1==str2double(test_FileNames{i}(1))
            correct=correct+1;
        end
    end
    rate=correct/length(test_FileNames);