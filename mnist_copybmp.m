%程序功能：从MNIST数据集60000图片的完整文件夹中复制每个类别图片a张到目标文件夹 

%目标文件夹地址
Dst_Path='C:\Users\llll\Desktop\类别相交数据集\印刷体\字母a\train\';
%所有类各取a张
a=10;

%读取训练集中所有图片名称
Path ='G:\研究生\工程资料\机器学习\数据集\印刷数字和字母\train\';   
File = dir(fullfile(Path,'*.bmp'));  
FileNames = {File.name}';  
 
%求各类图片数N=[N1,N2,...,N10]'
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

%复制图片
for i=1:10
    for j=1:N(i)
        %pic_num：当前为第几张图片
        pic_num=sum(M(1:i))+j;
        filename=[Path,File(pic_num).name];  
        copyfile(filename,Dst_Path);
    end
end
