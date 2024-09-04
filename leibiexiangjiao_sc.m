%����ཻ��������ģ��
    %% ���ò���
    %c1������Ƽ���c1*eye()��ȷ���ƿ���
    c1=0.001;

    %% ��ȡѵ����������ͼƬ
    train_Path = 'C:\Users\llll\Desktop\����ཻ���ݼ�\train\';   
    train_File = dir(fullfile(train_Path,'*.bmp'));  
    train_FileNames = {train_File.name}';           
    %% ��ϸ�����������N
    %00-ӡˢ��ĸ��01-ӡˢ���֣�10-��д��ĸ��11-��д����
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
    %% ��1����Pi=[P(1x);P(x1)]
    P1x=(N(3)+N(4))/sum(N);
    Px1=(N(2)+N(4))/sum(N);
    Pi=[P1x;Px1];
    %% ��2�����1=[��(1x),��(x1)]��784*2������2=[��(0x),��(x0)]��784*2��
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
    %% ��3�����1,��2(784*(784*2))
    variance1=zeros(784,784);
    variance2=zeros(784,784);
    for i=1:length(train_FileNames)
        num1=str2double(train_FileNames{i}(1));
        num2=str2double(train_FileNames{i}(2));
        Img=imread(strcat(train_Path,train_FileNames{i}));
        Img1=imresize(Img,[28,28]);
        Img2=im2bw(Img1);
        x=im2double(Img2(:));
        %var1IS0ָ��һ��Ԫ��Ϊ0�ġ��ۼӲ��֣�����ͬ��
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
    %% ��4������
    test_Path = 'C:\Users\llll\Desktop\����ཻ���ݼ�\test\';   
    test_File = dir(fullfile(test_Path,'*.bmp'));  
    test_FileNames = {test_File.name}'; 

    correct=0;
    for i=1:length(test_FileNames)
        Img=imread(strcat(test_Path,test_FileNames{i}));
        Img1=imresize(Img,[28,28]);
        Img2=im2bw(Img1);
        x=im2double(Img2(:));
        %��y
        y=zeros(2,1);
        P1Is1=Pi(1)*mvnpdf(x',average1(:,1)',variance1+c1*eye(784,784));
        P1Is0=(1-Pi(1))*mvnpdf(x',average1(:,2)',variance1+c1*eye(784,784));
        P2Is1=Pi(2)*mvnpdf(x',average2(:,1)',variance2+c1*eye(784,784));
        P2Is0=(1-Pi(2))*mvnpdf(x',average2(:,2)',variance2+c1*eye(784,784));
        y(1)=(P1Is1>=P1Is0);
        y(2)=(P2Is1>=P2Is0);
        %��t
        num1=str2double(test_FileNames{i}(1));
        num2=str2double(test_FileNames{i}(2));
        t=[num1;num2];
        %�ж�
        if sum(y==t)==2
            correct=correct+1;
        end
    end
    rate=correct/length(test_FileNames);