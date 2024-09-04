%by Lihuanlin 2022/11
%程序功能：对MNIST测试集做K均值聚类

%% 读取测试集样本
    Path = 'G:\研究生\实验资料\机器学习\数据集\手写数字MNIST\MNIST_bmp\test_img\';   
    File = dir(fullfile(Path,'*.bmp'));  
    FileNames = {File.name}';
    Xn=zeros(784,length(FileNames));
    Label=zeros(length(FileNames),1);
    for i=1:length(FileNames)
        Img=imread(strcat(Path,FileNames{i}));
        x=im2double(Img(:));
        Xn(:,i)=x;
        Label(i)=str2double(FileNames{i}(1));
    end
 
%% 求理想的聚类中心    
    
%% 初始化参数
    K=10;
    itnum=2000;
    u=rand(784,K);    
    for iterate=1:itnum    
        %% 固定u，确定每个样本所属类别
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
            u(:,i)=sum(Xk,2)/nk;
        end
    end

%% 测试：再固定u，确定每个样本所属类别，是否与标签一致
    dist_test=zeros(10,length(FileNames));
    for i=1:K
        uu=repmat(u(:,i),1,length(FileNames));
        dist_test(i,:)=sum((Xn-uu).^2);
    end
    z_test=zeros(length(FileNames),1);
    for i=1:length(FileNames)
        [value,pos]=min(dist_test(:,i));
        z_test(i)=pos-1;
    end  
    
    %类别=标签时为0，≠时不为
    Correct=sum(z_test==Label);
    rate=Correct/length(FileNames);