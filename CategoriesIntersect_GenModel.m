%类别相交——生成模型
    %% 设置参数
    %c1：矩阵∑加上c1*eye()，确保∑可逆
    c1=0.001;

    %% 读取训练集中所有图片
    train_Path = 'C:\Users\llll\Desktop\类别相交数据集\train\';   
    train_File = dir(fullfile(train_Path,'*.bmp'));  
    train_FileNames = {train_File.name}';           
    %% 求细分类别样本数N
    %00-印刷字母，01-印刷数字，10-手写字母，11-手写数字
    N=zeros(4,1);
    for i=1:length(train_FileNames)
        num1=str2double(train_FileNames{i}(1));
        num2=str2double(train_FileNames{i}(2));
        if num1==0&&num2==0
            N(1)=N(1)+1;
        end
        if num1==0&&num2==1
            N(2)=N(2)+1;
        end
        if num1==1&&num2==0
            N(3)=N(3)+1;
        end
        if num1==1&&num2==1
            N(4)=N(4)+1;
        end
    end
    %% 【1】求Pi=[P(1x);P(x1)]
    P1x=(N(3)+N(4))/sum(N);
    Px1=(N(2)+N(4))/sum(N);
    Pi=[P1x;Px1];
    %% 【2】求μ1=[μ(1x),μ(x1)]（784*2），μ2=[μ(0x),μ(x0)]（784*2）
    average1=zeros(784,2);
    average2=zeros(784,2);
    for i=1:length(train_FileNames)
        num1=str2double(train_FileNames{i}(1));
        num2=str2double(train_FileNames{i}(2));
        Img=imread(strcat(train_Path,train_FileNames{i}));
        Img1=imresize(Img,[28,28]);
        Img2=im2bw(Img1);
        x=im2double(Img2(:));
        if num1==0&&num2==0
            average2(:,1)=average2(:,1)+x/sum(N);
            average2(:,2)=average2(:,2)+x/sum(N);
        end
        if num1==0&&num2==1
            average2(:,1)=average2(:,1)+x/sum(N);
            average1(:,2)=average1(:,2)+x/sum(N);
        end
        if num1==1&&num2==0
            average1(:,1)=average1(:,1)+x/sum(N);
            average2(:,2)=average2(:,2)+x/sum(N);
        end
        if num1==1&&num2==1
            average1(:,1)=average1(:,1)+x/sum(N);
            average1(:,2)=average1(:,2)+x/sum(N);
        end
    end
    %% 【3】求∑1,∑2(784*(784*2))
    variance1=zeros(784,784);
    variance2=zeros(784,784);
    for i=1:length(train_FileNames)
        num1=str2double(train_FileNames{i}(1));
        num2=str2double(train_FileNames{i}(2));
        Img=imread(strcat(train_Path,train_FileNames{i}));
        Img1=imresize(Img,[28,28]);
        Img2=im2bw(Img1);
        x=im2double(Img2(:));
        %var1IS0指第一个元素为0的∑累加部分，其余同理
        var1IS0=0;
        var1IS1=0;
        var2IS0=0;
        var2IS1=0;
        if num1==0&&num2==0
            var1IS0=(x-average1(:,2))*(x-average1(:,2))'/sum(N);
            var2IS0=(x-average2(:,2))*(x-average2(:,2))'/sum(N);
        end
        if num1==0&&num2==1
            var1IS0=(x-average1(:,2))*(x-average1(:,2))'/sum(N);
            var2IS1=(x-average2(:,1))*(x-average2(:,1))'/sum(N);
        end
        if num1==1&&num2==0
            var1IS1=(x-average1(:,1))*(x-average1(:,1))'/sum(N);
            var2IS0=(x-average2(:,2))*(x-average2(:,2))'/sum(N);
        end
        if num1==1&&num2==1
            var1IS1=(x-average1(:,1))*(x-average1(:,1))'/sum(N);
            var2IS1=(x-average2(:,1))*(x-average2(:,1))'/sum(N);
        end
        variance1=variance1+var1IS0+var1IS1;
        variance2=variance2+var2IS0+var2IS1;
    end     
    %% 【4】测试
    test_Path = 'C:\Users\llll\Desktop\类别相交数据集\test\';   
    test_File = dir(fullfile(test_Path,'*.bmp'));  
    test_FileNames = {test_File.name}'; 

    correct=0;
    for i=1:length(test_FileNames)
        Img=imread(strcat(test_Path,test_FileNames{i}));
        Img1=imresize(Img,[28,28]);
        Img2=im2bw(Img1);
        x=im2double(Img2(:));
        %求y
        y=zeros(2,1);
        P1Is1=Pi(1)*mvnpdf(x',average1(:,1)',variance1+c1*eye(784,784));
        P1Is0=(1-Pi(1))*mvnpdf(x',average1(:,2)',variance1+c1*eye(784,784));
        P2Is1=Pi(2)*mvnpdf(x',average2(:,1)',variance2+c1*eye(784,784));
        P2Is0=(1-Pi(2))*mvnpdf(x',average2(:,2)',variance2+c1*eye(784,784));
        y(1)=(P1Is1>=P1Is0);
        y(2)=(P2Is1>=P2Is0);
        %求t
        num1=str2double(test_FileNames{i}(1));
        num2=str2double(test_FileNames{i}(2));
        t=[num1;num2];
        %判断
        if sum(y==t)==2
            correct=correct+1;
        end
    end
    rate=correct/length(test_FileNames);
