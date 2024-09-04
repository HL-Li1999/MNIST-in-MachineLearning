%����ģ��

%% ���ò���ֵ

    %���û�Ӱ��ʶ��׼ȷ�ʵĲ�����ֵ
    %c1��ͼƬ��ɫ����ֵ��Ϊ1*c1
    c1=1;
    %c2���Ʋ�Ϊ����������˼Ӿ���c2*eye()
    c2=0.1;
    
%% ѵ��

    %% ��1����ȡѵ����������ͼƬ
    train_Path = 'G:\�о���\ʵ������\����ѧϰ\���ݼ�\��д����MNIST\MNIST_bmp\train_img\';   
    train_File = dir(fullfile(train_Path,'*.bmp'));  
    train_FileNames = {train_File.name}';           
    
    %% ��2����Pc=[Pc1,Pc2,...,Pc10]'
    N=zeros(10,1);
    for i=1:length(train_FileNames)
        num=str2double(train_FileNames{i}(1));
        N(num+1)=N(num+1)+1;
    end
    Pc=N/sum(N);
    
    %% ��3�����=[��1,��2,...,��10]��784��10
    M=[0;N];
    x=zeros(784,1);
    average=zeros(784,10);
    %ѭ��ѵ��������ͼƬ
    for i=1:10
        for j=1:N(i)
            %��x�������ؾ���28*28ת��Ϊ����ʸ��784*1
            %pic_num����ǰΪ�ڼ���ͼƬ
            pic_num=sum(M(1:i))+j;           
            Img=imread(strcat(train_Path,train_FileNames{pic_num}));
            %����ֵ��uint8->double
            x=im2double(Img(:))*c1;           
            %���
            avr_kn=zeros(784,10);
            avr_kn(:,i)=x/N(i);
            average=average+avr_kn;             
        end
    end
    
    %% ��4����Э��������784��784
    variance=zeros(784,784);
    %ѭ��ѵ����������Ƭ
    for i=1:10
        for j=1:N(i)
            %����x�������ؾ���28*28ת��Ϊ����ʸ��784*1
            pic_num=sum(M(1:i))+j;
            Img=imread(strcat(train_Path,train_FileNames{pic_num}));
            x=im2double(Img(:))*c1;   
            %����� 
            variance=variance+(x-average(:,i))*(x-average(:,i))'/sum(N);
        end  
    end
        
%% ����    
    %% ��1����ȡ���Լ���������Ƭ
    test_Path = 'G:\�о���\ʵ������\����ѧϰ\���ݼ�\��д����MNIST\MNIST_bmp\test_img\';   
    test_File = dir(fullfile(test_Path,'*.bmp'));  
    test_FileNames = {test_File.name}';            
    
    %% ��2�����������õ��ĦС��̡��������ʶ��ͼƬ���ڸ����ĸ�˹����p(x,t)=[p(x,t1),...,p(x,t10)]'
    correct=0;
    %ѭ�����Լ�����ͼƬ
    for i=1:length(test_FileNames)
        %����x�������ؾ���28*28ת��Ϊ����ʸ��784*1      
        Img=imread(strcat(test_Path,test_FileNames{i}));
        x=im2double(Img(:))*c1;
        %����p(x,t)=[p(x,t1),p(x,t2),...,p(x,t10)]'
        p=zeros(10,1);
        for k=1:10
            %mvnpdfΪ���ά��˹����ֵ
            p(k)=Pc(k)*mvnpdf(x',average(:,k)',variance+c2*eye(784,784));
        end
        %���жϽ��
        [max_value,max_pos]=max(p);
        if (max_pos-1)==str2double(test_FileNames{i}(1))
            correct=correct+1;
        end
    end
    rate=correct/length(test_FileNames);
