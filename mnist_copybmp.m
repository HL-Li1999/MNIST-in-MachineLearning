%�����ܣ���MNIST���ݼ�60000ͼƬ�������ļ����и���ÿ�����ͼƬa�ŵ�Ŀ���ļ��� 

%Ŀ���ļ��е�ַ
Dst_Path='C:\Users\llll\Desktop\����ཻ���ݼ�\ӡˢ��\��ĸa\train\';
%�������ȡa��
a=10;

%��ȡѵ����������ͼƬ����
Path ='G:\�о���\��������\����ѧϰ\���ݼ�\ӡˢ���ֺ���ĸ\train\';   
File = dir(fullfile(Path,'*.bmp'));  
FileNames = {File.name}';  
 
%�����ͼƬ��N=[N1,N2,...,N10]'
N=zeros(62,1);
for i=1:length(FileNames)
%     num1=str2double(FileNames{i}(1));
%     num2=double(FileNames{i}(2));
%     if num2==95
%         num=num1;
%     end
%     if num2~=95
%         num2=num2-48;
%         num=num1*10+num2;
%     end
    num=str2double(FileNames{i}(1));
    N(num+1)=N(num+1)+1;
end
M=[0;N];

%����ͼƬ
for i=1:10
    for j=1:N(i)
        %pic_num����ǰΪ�ڼ���ͼƬ
        pic_num=sum(M(1:i))+j;
        filename=[Path,File(pic_num).name];  
        copyfile(filename,Dst_Path);
    end
end
