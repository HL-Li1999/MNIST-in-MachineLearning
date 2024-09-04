%生成模型

%% 设置参数值

    %设置会影响识别准确率的参数的值
    %c1：图片白色像素值设为1*c1
    c1=1;
    %c2：∑不为正定矩阵，因此加矩阵c2*eye()
    c2=0.1;
    
%% 训练

    %% 【1】读取训练集中所有图片
    train_Path = 'G:\研究生\实验资料\机器学习\数据集\手写数字MNIST\MNIST_bmp\train_img\';   
    train_File = dir(fullfile(train_Path,'*.bmp'));  
    train_FileNames = {train_File.name}';           
    
    %% 【2】求Pc=[Pc1,Pc2,...,Pc10]'
    N=zeros(10,1);
    for i=1:length(train_FileNames)
        num=str2double(train_FileNames{i}(1));
        N(num+1)=N(num+1)+1;
    end
    Pc=N/sum(N);
    
    %% 【3】求μ=[μ1,μ2,...,μ10]，784×10
    M=[0;N];
    x=zeros(784,1);
    average=zeros(784,10);
    %循环训练集所有图片
    for i=1:10
        for j=1:N(i)
            %求x：将像素矩阵28*28转换为像素矢量784*1
            %pic_num：当前为第几张图片
            pic_num=sum(M(1:i))+j;           
            Img=imread(strcat(train_Path,train_FileNames{pic_num}));
            %像素值从uint8->double
            x=im2double(Img(:))*c1;           
            %求μ
            avr_kn=zeros(784,10);
            avr_kn(:,i)=x/N(i);
            average=average+avr_kn;             
        end
    end
    
    %% 【4】求协方差矩阵∑784×784
    variance=zeros(784,784);
    %循环训练集所有照片
    for i=1:10
        for j=1:N(i)
            %①求x：将像素矩阵28*28转换为像素矢量784*1
            pic_num=sum(M(1:i))+j;
            Img=imread(strcat(train_Path,train_FileNames{pic_num}));
            x=im2double(Img(:))*c1;   
            %②求∑ 
            variance=variance+(x-average(:,i))*(x-average(:,i))'/sum(N);
        end  
    end
        
%% 测试    
    %% 【1】读取测试集中所有照片
    test_Path = 'G:\研究生\实验资料\机器学习\数据集\手写数字MNIST\MNIST_bmp\test_img\';   
    test_File = dir(fullfile(test_Path,'*.bmp'));  
    test_FileNames = {test_File.name}';            
    
    %% 【2】代入上述得到的π、μ、∑求出待识别图片对于各类别的高斯概率p(x,t)=[p(x,t1),...,p(x,t10)]'
    correct=0;
    %循环测试集所有图片
    for i=1:length(test_FileNames)
        %①求x：将像素矩阵28*28转换为像素矢量784*1      
        Img=imread(strcat(test_Path,test_FileNames{i}));
        x=im2double(Img(:))*c1;
        %②求p(x,t)=[p(x,t1),p(x,t2),...,p(x,t10)]'
        p=zeros(10,1);
        for k=1:10
            %mvnpdf为求多维高斯函数值
            p(k)=Pc(k)*mvnpdf(x',average(:,k)',variance+c2*eye(784,784));
        end
        %③判断结果
        [max_value,max_pos]=max(p);
        if (max_pos-1)==str2double(test_FileNames{i}(1))
            correct=correct+1;
        end
    end
    rate=correct/length(test_FileNames);
