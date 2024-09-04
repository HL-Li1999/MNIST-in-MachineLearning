%by Lihuanlin 2022/12
%程序功能：对MNIST训练集作BMM聚类，用测试集测试

%% 训练
%% 取pik，miuk初始值
    p=rand(10,1);i=sum(p);pi=p/i;
    miu=0.5*ones(784,10);
    y=zeros(10,60000);
    PXn_miu=zeros(10,60000);
    
%% 读取训练集样本
    Path = 'G:\研究生\实验资料\机器学习\数据集\手写数字MNIST\MNIST_bmp\train_img\';   
    File = dir(fullfile(Path,'*.bmp'));  
    FileNames = {File.name}';
    %%
    Xn=zeros(784,length(FileNames));
    for i=1:length(FileNames)
        Img=imread(strcat(Path,FileNames{i}));
        x=im2double(Img(:));
        Xn(:,i)=x;
    end 
    
    %%
    itnum=15;
    Ex=zeros(itnum,1); 
    for iterate=1:itnum
        %% E step
%         Xn=ones(784,60000);
%         miu=0.5*ones(784,10);
%         pi=ones(10,1);
        for k=1:10
            for j=1:10
                PXn_miu(j,:)=prod(miu(:,j).^Xn).*prod((ones(784,1)-miu(:,j)).^(ones(784,60000)-Xn));
            end
            y(k,:)=pi(k)*PXn_miu(k,:)./sum(repmat(pi,1,60000).*PXn_miu);
        end
        %% M step
        for k=1:10
            N=sum(y,2);
            y_znk=repmat(y(k,:),784,1);
            miu(:,k)=sum(y_znk.*Xn,2)/N(k);
            pi(k)=N(k)/60000;
        end
        Ex(iterate)=sum(log(sum(repmat(pi,1,60000).*PXn_miu)));
%         if iterate~=1&&Ex(iterate)==Ex(iterate)
%             break;
%         end
    end
    [~,pos]=max(y);
    %%
%     for i=1:length(pos)
%         if pos(i)==1
%             filename=[Path,File(i).name];
%             Dst_Path='C:\Users\llll\Desktop\1\1\';
%             copyfile(filename,Dst_Path);
%         end
%     end
    %%
    maxnum1=zeros(10,1); %聚类得到的同一类为同一数字的最大占比
    maxk=zeros(10,1); %miu(k)对应手写数字maxk(k)
    for k=1:10
        num=zeros(10,1); %聚类得到的同一类中各个手写数字的数量
        for i=1:length(pos)
            if pos(i)==k
                num(str2double(FileNames{i}(1))+1)=num(str2double(FileNames{i}(1))+1)+1;
            end
        end   
        [maxnum11,maxk(k)]=max(num);
        maxnum1(k)=maxnum11/sum(num);
    end
%% 测试
%% 依次读取测试集样本
    Xn=zeros(784,10000);
    testPath = 'G:\研究生\实验资料\机器学习\数据集\手写数字MNIST\MNIST_bmp\test_img\';   
    testFile = dir(fullfile(testPath,'*.bmp'));  
    testFileNames = {testFile.name}';
    for i=1:length(testFileNames)
        Img=imread(strcat(testPath,testFileNames{i}));
        x=im2double(Img(:));
        Xn(:,i)=x;
    end 
    %%
    y=zeros(10,10000);
    PXn_miu=zeros(10,10000);
    for k=1:10
        for j=1:10
            PXn_miu(j,:)=prod(miu(:,j).^Xn).*prod((ones(784,10000)-miu(:,j)).^(ones(784,10000)-Xn));
        end
        y(k,:)=pi(k)*PXn_miu(k,:)./sum(repmat(pi,1,10000).*PXn_miu);
    end
    [~,testk]=max(y);
    %%
    correct=0;
    for i=1:length(testk)
        if maxk(testk(i))==(str2double(testFileNames{i}(1))+1)
            correct=correct+1;
        end
    end
    rate=correct/10000;            
    %%
%     num2=zeros(10,1);
%     for k=1:10
%         for i=1:length(pos2)
%             if pos2(i)==k
%                 num2(str2double(testFileNames{i}(1))+1)=num2(str2double(testFileNames{i}(1))+1)+1;
%             end
%         end   
%         [maxnum11,~]=max(num2);
%         pos2(k)=pos22/sum(num2);
%     end
%%
    plot(1:itnum,Ex);