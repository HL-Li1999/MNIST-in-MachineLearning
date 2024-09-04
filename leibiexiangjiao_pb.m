%类别相交——判别模型
%% 调节参数
%设置会影响识别准确率的参数的值
%c1：每个白像素的元素值为1*c1
c1=1;
%c2：w初始值为c2*ones()
c2=1; 
%c3：H不正定因此加上c3*max(abs(diag(H)))*eye()
c3=1e-6;
%% 初始化
    %读取训练集中所有图片
    train_Path ='C:\Users\llll\Desktop\类别相交数据集\train\';   
    train_File = dir(fullfile(train_Path,'*.bmp'));  
    train_FileNames = {train_File.name}';            
    %求细分类别样本数N
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
%% 训练
    itnum=1000;
    wn_1=zeros(785,itnum+1);wn_2=zeros(785,itnum+1);
    wn_1(:,1)=c2*ones(785,1);wn_2(:,1)=c2*ones(785,1);
    Ewn_1=zeros(itnum,1);Ewn_2=zeros(itnum,1);
    Gradient_1=zeros(785,itnum);Gradient_2=zeros(785,itnum);
    for iterate=1:itnum
        wnow_1=wn_1(:,iterate); 
        wnow_2=wn_2(:,iterate);       
        %累加量初始值
        Ew_1i=zeros(sum(N),1);Ew_2i=zeros(sum(N),1);
        Gradient_1i=zeros(785,sum(N));Gradient_2i=zeros(785,sum(N));
        H_1=zeros(785,785);H_2=zeros(785,785);
        for i=1:sum(N)
           %% 【1】求t=[t1;t2]
            num1=str2double(train_FileNames{i}(1));
            num2=str2double(train_FileNames{i}(2));
            t=[num1;num2];
           %% 【2】求y=[y1;y2]
            Img=imread(strcat(train_Path,train_FileNames{i}));
            Img1=imresize(Img,[28,28]);
            Img2=im2bw(Img1);
            x=im2double(Img2(:))*c1;
            x_feature=[1;x]; 
            %a=[a1;a2]
            %限制-a的最大值<=100，避免-a太大导致exp(-a)出现Inf值
            a=[wnow_1,wnow_2]'*x_feature; 
            if min(a)<-100
                b=(-100-min(a))*ones(2,1);
                a=a+b;
            end
            %y=[y1,y2]，注意用点除否则会变成行向量且值不正确
            y=[1;1]./(1+exp(-a));
           %% 【3】求E1(w)、E2(w)
            %限制y∈[0.01,0.99]，避免y出现0导致log(y)或log(1-y)为-Inf值
            if y(1)<0.01
                y(1)=0.01;
            end
            if y(1)>0.99
                y(1)=0.99;
            end
            if y(2)<0.01
                y(2)=0.01;
            end
            if y(2)>0.99
                y(2)=0.99;
            end
            Ew_1i(i)=-(t(1)*log(y(1))+(1-t(1))*log(1-y(1)));
            Ew_2i(i)=-(t(2)*log(y(2))+(1-t(2))*log(1-y(2)));
           %% 【4】求Gradient_1(w)、Gradient_2(w)
            Gradient_1i(:,i)=(y(1)-t(1))*x_feature;
            Gradient_2i(:,i)=(y(2)-t(2))*x_feature;
           %% 【5】求H(w)=H1(w)+H2(w)
            H_1=H_1+(y(1)*(1-y(1)))*(x_feature*x_feature');
            H_2=H_2+(y(2)*(1-y(2)))*(x_feature*x_feature');
        end
        Ew_1=sum(Ew_1i);
        Ew_2=sum(Ew_2i);
        Ewn_1(iterate)=Ew_1;
        Ewn_2(iterate)=Ew_2;
        Gradient_1=sum(Gradient_1i,2);
        Gradient_2=sum(Gradient_2i,2);
       %% 【6】求w_next
        c3_it=c3;
        v=2;
        while 1
            %注意加绝对值，否则会出现对角元素<=0，最大值为0的情况，这样c3多大都没用
            H1_rec=H1+c3_it*max(abs(diag(H1)))*eye(785,785);
            if det(H1_rec)<=0
                c3_it=c3_it*v;
                v=v*2;
            else
                break;
            end
        end
        c3_it=c3;
        v=2;
        while 1
            H2_rec=H2+c3_it*max(abs(diag(H2)))*eye(785,785);
            if det(H2_rec)<=0
                c3_it=c3_it*v;
                v=v*2;
            else
                break;
            end
        end
        wnext_1=wnow_1-H1_rec\Gradient_1;
        wnext_2=wnow_2-H2_rec\Gradient_2;
        wn_1(:,iterate+1)=wnext_1;
        wn_2(:,iterate+1)=wnext_2;
    end
%% 测试
    test_Path = 'C:\Users\llll\Desktop\类别相交数据集\test\';   
    test_File = dir(fullfile(test_Path,'*.bmp'));  
    test_FileNames = {test_File.name}'; 

    correct=0;
    for i=1:length(test_FileNames)
        test_Img=imread(strcat(test_Path,test_FileNames{i}));
        test_Img1=imresize(test_Img,[28,28]);
        test_Img2=im2bw(test_Img1);
        test_x=im2double(test_Img2(:))*c1;
        testx_feature=[1;test_x];
        %求y
        testw_1=wn_1(:,201);
        testw_2=wn_2(:,201);
        testa_1=testw_1'*testx_feature;
        testa_2=testw_2'*testx_feature;
        test_y=zeros(2,1);
        if testa_1>=0
            test_y(1)=1;
        end
        if testa_2>=0
            test_y(2)=1;
        end
        %求t
        test_num1=str2double(test_FileNames{i}(1));
        test_num2=str2double(test_FileNames{i}(2));
        test_t=[test_num1;test_num2];
        %判断
        if sum(test_y==test_t)==2
            correct=correct+1;
        end
    end
    rate=correct/length(test_FileNames);
%% 画图
    I=1:itnum;
    plot(I,Ewn_1);
    hold on;
    plot(I,Ewn_2);
    text(5,Ewn_1(50),'E1(w)');
    text(5,Ewn_2(50),'E2(w)');
    