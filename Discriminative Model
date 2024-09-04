%判别模型

%% 设置参数值
%设置会影响识别准确率的参数的值
%c1：每个白像素的元素值为1*c1
c1=1;
%c2：w初始值为c2*ones() 
c2=0.01;
%c3：H不可逆时加上c3*max(abs(diag(Ew11)))*eye()
c3=200;
%% 训练
    %% 读取训练集中所有图片
    train_Path ='G:\研究生\机器学习\数据集\手写数字MNIST\MNIST_bmp\train_img\';   
    train_File = dir(fullfile(train_Path,'*.bmp'));  
    train_FileNames = {train_File.name}'; 
    %% 求各类图片数N=[N1,N2,...,N10]'
    N=zeros(10,1);
    for i=1:length(train_FileNames)
        num=str2double(train_FileNames{i}(1)); 
        N(num+1)=N(num+1)+1;
    end   
    %% 循环迭代，求E(w)极小值点
    itnum=10;  %最大迭代次数
    wn=zeros(7850,itnum+1);
    %wn=load('wn_panbie(1).txt');
    wn(:,1)=c2*ones(7850,1);
    Ewn=zeros(itnum,1); 
    for iterate=1:itnum  
        %W_now=[w1;...;w10](785*10) 
        w_now=wn(:,iterate);
        W_now=reshape(w_now,785,10); 
        %累加量初始值
        Ew=0;
        Gradient=zeros(7850,1);
        H=zeros(7850,7850);
        %% 循环所有图片
        for i=1:sum(N)
            %% 求Ф、t、y 
            Img=imread(strcat(train_Path,train_FileNames{i}));    
            x=im2double(Img(:))*c1;
            x_feature=[1;x];       %Ф=[1,x1,...,x784]'
            t=zeros(10,1);
            t(str2double(train_FileNames{i}(1))+1,1)=1;  %t=[t1,...,t10]'
            a=W_now'*x_feature;
            %限制a的最大值<=100，避免a太大导致exp(a)出现Inf值
            if max(a)>100
                b=(max(a)-100)*ones(10,1);
                a=a-b;
            end
            y=exp(a)/sum(exp(a));   %求y=[y1,...,y10]'
            %限制y的最小值>=0.01，避免y出现0导致log(y)为-Inf值，始终满足sum(y)=1
            if min(y)==0
                [max_value,max_pos]=max(y);
                y=y+0.01*ones(10,1);
                y(max_pos)=max_value-0.1;
            end
           %% 求Ew、梯度、Hessian矩阵
            Ew=Ew-t'*log(y);  %Ew
            features=repmat(x_feature,10,1);
            Y_T1=repmat(y-t,1,785);
            Y_T2=Y_T1';
            Y_T3=Y_T2(:);
            Gradient=Gradient+Y_T3.*features; %梯度              
            x_ff=x_feature*x_feature';
            X_ff=repmat(x_ff,10,10);  
            coefficient=y*y'-diag(y);
            Coefficient=kron(coefficient,ones(785,785));
            H=H+X_ff.*Coefficient; %Hessian矩阵
        end       
       %% 确保H正定
        %注意加绝对值，否则会出现对角元素<=0，最大值为0的情况，这样c3多大都没用
        H1=H+c3*max(abs(diag(H)))*eye(7850,7850);
        w_next=w_now-H1\Gradient;   
        wn(:,iterate+1)=w_next;
        Ewn(iterate)=Ew;
%         if iterate~=1
%             if abs(Ewn(iterate)-Ewn(iterate-1))<1e-3
%                 break;
%             end
%         end        
    end       
%% 保存矩阵、向量为txt
%     [m, n] = size(wn);
%     txtname='wn_panbie(1).txt';
%     fid=fopen(txtname, 'wt');
%     for i = 1 : m
%         fprintf(fid, '%g\t', wn(i, :));
%         fprintf(fid, '\n');
%     end    
%% 测试
    %【1】读取测试集中所有照片
    test_Path = 'G:\研究生\工程资料\机器学习\数据集\手写数字MNIST\MNIST_bmp\test_img\';   % 设置数据存放的文件夹路径
    test_File = dir(fullfile(test_Path,'*.bmp'));  % 显示文件夹下所有符合后缀名为.txt文件的完整信息
    test_FileNames = {test_File.name}';            % 提取符合后缀名为.txt的所有文件的文件名，转换为n行1列    
    %【2】代入训练得到的w，测试结果
    %wn=load('w_next.txt');
    correct=0;
    %循环测试集所有照片  
    for i=1:length(test_FileNames)
        %①求Ф(x)
        test_Img=imread(strcat(test_Path,test_FileNames{i}));
        test_x=im2double(test_Img(:))*c1;
        testx_feature=[1;test_x];
        %②求y
        test_w=wn(:,iterate+1);
        test_W=reshape(test_w,785,10);
        test_a=test_W'*testx_feature;
        %限制test_a的最大值<=100，避免test_a太大导致exp(test_a)出现Inf值
        if max(test_a)>100
           test_b=(max(test_a)-100)*ones(10,1);
           test_a=test_a-test_b;
        end
        test_y=exp(test_a)/sum(exp(test_a));
        %③判断结果
        [max_value,max_pos]=max(test_y);
        if max_pos-1==str2double(test_FileNames{i}(1))
            correct=correct+1;
        end
    end
    rate=correct/length(test_FileNames);   
%% 画图
    k=7;
    I=1:k;
    Ewn=1.0e+05*[1.3816;1.3241;1.2779;1.2352;1.1949;1.1568;1.1207];
    figure(1);
    plot(I,Ewn(1:k));
    text(1,Ewn(1),'E(w)');
    figure(2);
    RATE=[0.098;0.4001;0.5831;0.6715;0.71;0.7306;0.7774;0.7816];
    plot(I,RATE(1:k));
    text(1,RATE(1),'RATE');
