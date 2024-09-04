%by Lihuanlin 2022/11
%�����ܣ���MNIST���Լ���K��ֵ����

%% ��ȡ���Լ�����
    Path = 'G:\�о���\ʵ������\����ѧϰ\���ݼ�\��д����MNIST\MNIST_bmp\test_img\';   
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
 
%% ������ľ�������    
    
%% ��ʼ������
    K=10;
    itnum=2000;
    u=rand(784,K);    
    for iterate=1:itnum    
        %% �̶�u��ȷ��ÿ�������������
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
        %% �̶�z����u
        for i=1:K 
            zi=repmat(z(i,:),784,1);
            Xk=zi.*Xn;
            nk=sum(z(i,:));
            u(:,i)=sum(Xk,2)/nk;
        end
    end

%% ���ԣ��ٹ̶�u��ȷ��ÿ��������������Ƿ����ǩһ��
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
    
    %���=��ǩʱΪ0����ʱ��Ϊ
    Correct=sum(z_test==Label);
    rate=Correct/length(FileNames);