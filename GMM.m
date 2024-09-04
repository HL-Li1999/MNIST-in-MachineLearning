%by Lihuanlin 2022/12
%�����ܣ���MNIST���Լ���GMM����

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

%% ��K=10ʱ�������������
    M=[0;N];
    Average=zeros(784,10);
    for i=1:10
        NUM1=sum(M(1:i))+1;
        NUM2=sum(M(1:i+1));
        Average(:,i)=sum(Xn(:,NUM1:NUM2),2)/N(i);
    end 
%% ��ʼ������
    K=10;
    itnum=5;
    p=rand(10,1);i=sum(p);pi=p/i;
    u=rand(784,10);
    Sigma=u*u'+0.1*eye(784,784);
    Sigma=[Sigma,Sigma,Sigma,Sigma,Sigma,Sigma,Sigma,Sigma,Sigma,Sigma];
    Z=zeros(10,length(FileNames));
    F=zeros(itnum,1);
    dist_u=zeros(itnum,1);
    dist_min=zeros(itnum,1);
    index=zeros(10,itnum); 
    
    
%% 
     for iterate=1:itnum
        %% �̶�pi��u��Sigma����y(zkn)��y�Ĵ�СΪ(10*10000)��(k*n)
        u0=u;
        Nx=zeros(10,length(FileNames));
        for k=1:10
            Sigmak=Sigma(:,784*(k-1)+1:784*k);
            Nx(k,:)=mvnpdf(Xn',u(:,k)',Sigmak);
        end
        %����Nxĳһ������Ϊ0�����y��ĸΪ0
        ZeroCol=find(sum(Nx)==0);
        for i=1:length(ZeroCol)
            Nx(:,ZeroCol(i))=ones(10,1);
        end
        %����Nx̫С����y��ĸΪ0
        Nx=Nx*1.0e+230;
        y=repmat(pi,1,10000).*Nx./sum(repmat(pi,1,10000).*Nx);
        %% �������Ȼ����
        F(iterate)=sum(log(sum(repmat(pi,1,length(FileNames)).*y)));
        %% �̶�y(zkn)����Nk��pi��u��Sigma
        Nk=sum(y,2);
        pi=Nk/length(FileNames);
        for k=1:10 
            Sigmak=zeros(784,784);
            uk=zeros(784,1);
            for n=1:length(FileNames)
                uk=uk+y(k,n)/Nk(k)*Xn(:,n);
                Sigmak=Sigmak+y(k,n)/Nk(k)*(Xn(:,n)-u(:,k))*(Xn(:,n)-u(:,k))'+0.1*eye(784,784);
            end
            u(:,k)=uk;
            Sigma(:,784*(k-1)+1:784*k)=Sigmak;
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
     
     %%
     i=1:itnum;
     figure(1)
     plot(i',F);
     title('������Ȼ����ֵ');
     figure(2)
     plot(i',dist_u);
     title('�������ĵ�ÿ�����ƶ�����');
     figure(3)
     plot(i',dist_min);
     title('��������������������ĵ��������');   
        
    