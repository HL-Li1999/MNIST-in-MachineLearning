%by Lihuanlin 2022/11
%������;����K����ڽ�����������ģ�ͣ�������MNIST����

%% �趨����
    %��ȡ�۲�����ڽ���k��������
    k=3;
%% ��ȡѵ����������ͼƬ
    train_Path = 'G:\�о���\ʵ������\����ѧϰ\���ݼ�\��д����MNIST\MNIST_bmp\train_img\';   
    train_File = dir(fullfile(train_Path,'*.bmp'));  
    train_FileNames = {train_File.name}';    
%% Ԥ����������ͼƬ��
    N=zeros(10,1);
    for i=1:length(train_FileNames)
        num=str2double(train_FileNames{i}(1));
        N(num+1)=N(num+1)+1;
    end
    M=[0;N];
%% ��ȡ���Լ�����������
    test_Path = 'G:\�о���\ʵ������\����ѧϰ\���ݼ�\��д����MNIST\MNIST_bmp\test_img\';   
    test_File = dir(fullfile(test_Path,'*.bmp'));  
    test_FileNames = {test_File.name}';    
%% K�����
    %% ѵ������������ͼƬ��������
    Xn=zeros(784,60000);
    for i=1:length(train_FileNames)
        Img=imread(strcat(train_Path,train_FileNames{i}));
        x=im2double(Img(:)); 
        Xn(:,i)=x;
    end
    %% ���Լ�����һ��ͼƬ���۲��
    correct=0;
    for i=1:length(test_FileNames)
        Img=imread(strcat(train_Path,test_FileNames{i}));
        x_test=im2double(Img(:));
        [idx,id]= knnsearch(Xn',x_test','k',k);
        %��K����ڷ����ж�
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