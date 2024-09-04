%by Lihuanlin 2022/11
%�����ܣ���MNIST���Լ���K��ֵ����

%% ��ȡ���Լ�����
    Path = 'G:\�о���\ʵ������\����ѧϰ\���ݼ�\��д����MNIST\MNIST_bmp\test_img\';   
    File = dir(fullfile(Path,'*.bmp'));  
    FileNames = {File.name}';
    Xn=zeros(784,length(FileNames));
    N=zeros(10,1);
    for i=1:length(FileNames)
        Img=imread(strcat(Path,FileNames{i}));
        x=im2double(Img(:));
        Xn(:,i)=x;
        num=str2double(FileNames{i}(1))+1;
        N(num)= N(num)+1;
    end 
%% ��ʼ������
    K=10;
    itnum=100;
    u=rand(784,K); 
    dist_u=zeros(itnum,1);
    dist_min=zeros(itnum,1);
    index=zeros(10,itnum);    
%% ��K=10ʱ�������������
    M=[0;N];
    Average=zeros(784,10);
    for i=1:10
        NUM1=sum(M(1:i))+1;
        NUM2=sum(M(1:i+1));
        Average(:,i)=sum(Xn(:,NUM1:NUM2),2)/N(i);
    end
    %% ����
    for iterate=1:itnum    
        %% �̶�u��ȷ��ÿ�������������z
        u0=u;
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
            if nk==0
                nk=1;
            end
            u(:,i)=sum(Xk,2)/nk;
        end
        %% �������ĵ�ÿ�����ƶ�����
        dist_u(iterate)=sum(sqrt(sum((u-u0).^2)));       
        %% ��������������������ĵ��������
        Dist_min=zeros(10,1);
        for a=1:10
            %��������ʵ�ʾ�����������������������(�����ظ�)
            [idx,id]=knnsearch(Average',u(:,a)','k',1);
            Dist_min(a)=sum(sqrt(sum((Average(:,idx)-u(:,a)).^2)));
            %���Բ鿴��ʵ�ʾ�����������������������
            index(a,iterate)=idx;
        end
        dist_min(iterate)=sum(Dist_min);                       
    end  
    %% ��ͼ
    i=1:itnum;
    figure(1)
    plot(i',dist_u);
    title('�������ĵ�ÿ�����ƶ�����');
    figure(2)
    plot(i',dist_min);
    title('��������������������ĵ��������');