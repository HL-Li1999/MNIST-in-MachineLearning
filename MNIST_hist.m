%by Lihuanlin 2022/11
%�����ܣ���ȡMNISTѵ�������������������������Ӧ���ĵľ����ֱ��ͼ

%% ���ò���
    %ֱ��ͼ��bin�ĸ���
    numOfBins = 10;
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
    M=[0;N];    
%% ����ѵ��������������
    Xn=zeros(784,60000);
    for i=1:length(train_FileNames)
        Img=imread(strcat(train_Path,train_FileNames{i}));
        x=im2double(Img(:)); 
        Xn(:,i)=x;
    end    
%% ����ѵ��������λ��
    Average=zeros(784,10);
    for i=1:10
        COL1=sum(M(1:i))+1;
        COL2=sum(M(1:i+1));
        Average(:,i)=sum(Xn(:,COL1:COL2),2)/N(i);
    end
%     for i=1:10
%         AveragePic=reshape(Average(:,i),28,28);
%         mat2gray(AveragePic);
%         figure(i);
%         imshow(AveragePic);
%         title(num2str(i));
%     end
%% �������������Ӧ���ĵľ���
    dist=zeros(60000,1);
    for i=1:10
        for j=1:N(i)
            picnum=sum(M(1:i))+j;
            dist(picnum)=sqrt(sum((Xn(:,picnum)-Average(:,i)).^2));
        end
    end    
%% ��ʾֱ��ͼ
     hold off
    for i=10:10
        NUM1=sum(M(1:i))+1;
        NUM2=sum(M(1:i+1));
        [histFreq, histXout] = hist(dist(NUM1:NUM2), numOfBins);
        binWidth = histXout(2)-histXout(1);
        figure(i+10);
        bar(histXout, histFreq);       
        xlabel('distance');
        ylabel('K(distance)');
%         hold on

    end
%% ��ֱ��ͼ����ʾ��˹����
%     for i=1:10
%         figure(i+10);
%         x=4:14;
%         y=normpdf(x,7,1);
%         plot(x,y);
%     end    