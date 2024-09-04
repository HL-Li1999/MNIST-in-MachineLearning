%by Lihuanlin 2022/11
%������;���ú��ܶȹ��ƽ�����������ģ�ͣ�������MNIST���ݼ�

%% �趨����
    %ȡƽ������h��ֵ
    h=1;
%% ��ȡѵ��������������
    train_Path = 'G:\�о���\ʵ������\����ѧϰ\���ݼ�\��д����MNIST\MNIST_bmp\train_img\';   
    train_File = dir(fullfile(train_Path,'*.bmp'));  
    train_FileNames = {train_File.name}';    
%% Ԥ����������������
    N=zeros(10,1);
    for i=1:length(train_FileNames)
        num=str2double(train_FileNames{i}(1));
        N(num+1)=N(num+1)+1;
    end    
%% ��ȡ���Լ�����������
    %��������ÿһ��txt�ļ�Ϊ���г�����ͬ��������ľ���
    test_Path = 'G:\�о���\ʵ������\����ѧϰ\���ݼ�\��д����MNIST\MNIST_bmp\test_img\';   
    test_File = dir(fullfile(test_Path,'*.bmp'));  
    test_FileNames = {test_File.name}';
    
%% ���ܶȹ���
    %% ����ѵ��������������
    Xn=zeros(784,60000);
    for i=1:length(train_FileNames)
        Img=imread(strcat(train_Path,train_FileNames{i}));
        x=im2double(Img(:)); 
        Xn(:,i)=x;
    end
    %% ���Լ�����һ��ͼƬ�����۲�Ŀ��
    correct=0;
    M=[0;N];
    for i=1:length(test_FileNames)
        Img=imread(strcat(test_Path,test_FileNames{i}));
        x=im2double(Img(:)); 
        %���ú��ܶȹ�����p(x|Ck)
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
        %% ��P(Ck|x)    
        pCk=px.*N;
        %�ж�p(Ck|x_test)�����ֵ��Ϊ������
        [max_value,max_pos]=max(pCk);
        if max_pos-1==str2double(test_FileNames{i}(1))
            correct=correct+1;
        end
    end
    rate=correct/length(test_FileNames);
            