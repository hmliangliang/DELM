tic;
data=krkopt;%数据集
t=0.05;
winsize=1000;%滑动窗口的大小
col=size(data,2);%数据的列数
%数据进行标准化
tempdata=zscore(data(:,1:(col-1)));
data=[tempdata,data(:,col)];
clear tempdata;%清除临时变量
constant=5;
fun='sigmoid';
fun2='sigmoid';
accuracy=[];%保存每一次分类结果的准确率
%初始化结果
num1=[];%保存第一个隐含层节点的数目
num2=[];%保存第二个隐含层节点的数目
N1=5*col;%第一个隐含层节点的数目
N2=5*col;%第二个隐含层节点的数目
C=0.003;
p=0;
pmin=1;
smin=1;
c=0;%记录测试的
qnum=max(data(:,col));%类标签的最大值
s=sqrt(p*(1-p)/(c+1));
for i=1:size(data,1)
    if (i>=2*winsize)&&(mod(i,2*winsize)==0)
        traindata=data((i-2*winsize+1):(i-winsize),1:(col-1));%获取训练集
        traintarget=data((i-2*winsize+1):(i-winsize),col);%获取训练目标数据类标签
        testdata=data((i-winsize+1):i,1:(col-1));%获取测试集
        testtarget=data((i-winsize+1):i,col);%获取测试数据类标签
        %将训练集的目标值进行0-1化
        c=c+1;%第一次测试
        mytarget=zeros(size(traindata,1),qnum);%mytarget为训练集的测试数据的类标签0-1化结果
        for pp=1:size(traintarget,1)
            mytarget(pp,traintarget(pp,1))=1;
        end
        if (p+s)<=(pmin+2*smin)&&(p<t)%说明当前数据正常
           %初始化第一个隐含层节点的参数取值
           %disp('正常！');
           a1=rand((col-1),N1);
           b1=rand(1,1);
           temH=traindata*a1+b1;%计算矩阵的乘积
           %计算隐含层节点的输出值
           if strcmp(fun,'sigmoid')==1
              H1=1./exp(-(temH));
           elseif strcmp(fun,'radbas')==1
              H1=radbas(temH);
           elseif strcmp(fun,'hardlim')==1
              H1=double(hardlim(temH));
           elseif strcmp(fun,'sine')==1
              H1=sin(temH);
           end
           [H1,SP]=mapminmax(H1,0,1);
           %生成对角矩阵
           d=ones(1,N1);
           I=diag(d);
           beta1=(I./C+H1'*H1)\H1'*mytarget;
           K1=I./C+H1'*H1;
           %进入第一层隐含层在线学习步骤
           K1=K1+H1'*H1;
           beta1=beta1+K1\H1'*(mytarget-H1*beta1);
           H1=H1*beta1;
           [H1,SP]=mapminmax(H1,0,1);
           %初始化第二个隐含层节的参数
           %=zscore(H1);
           a2=rand(qnum,N2);
           b2=rand(1,1);
           temH2=H1*a2+b2;
           %计算隐含层节点的输出值
           if strcmp(fun2,'sigmoid')==1
              H2=1./exp(-(temH2));
           elseif strcmp(fun2,'radbas')==1
              H2=radbas(temH2);
           elseif strcmp(fun2,'hardlim')==1
              H2=double(hardlim(temH2));
           elseif strcmp(fun2,'sine')==1
              H2=sin(temH2);
           end
           [H2,SP]=mapminmax(H2,0,1);
           %计算参数
           d=ones(1,N2);
           I=diag(d);
           beta2=(I./C+H2'*H2)\H2'*mytarget;
           K2=I./C+H2'*H2;
           %进入第二层隐含层在线学习步骤
           K2=K2+H2'*H2;
           beta2=beta2+K2\H2'*(mytarget-H2*beta2);
        elseif (p+s>pmin+2*smin)&&(p+s<pmin+3*smin)%达到警告水平
            %disp('警告！');
            %更新隐含层神经元的数目
            N2=N1;
            N1=min(winsize,N1+floor(constant/(p+0.001)));
           %初始化第一个隐含层节点的参数取值
           a1=rand((col-1),N1);
           b1=rand(1,1);
           temH=traindata*a1+b1;%计算矩阵的乘积
           %计算隐含层节点的输出值
           if strcmp(fun,'sigmoid')==1
              H1=1./exp(-(temH));
           elseif strcmp(fun,'radbas')==1
              H1=radbas(temH);
           elseif strcmp(fun,'hardlim')==1
              H1=double(hardlim(temH));
           elseif strcmp(fun,'sine')==1
              H1=sin(temH);
           end
           [H1,SP]=mapminmax(H1,0,1);
           %生成对角矩阵
           d=ones(1,N1);
           I=diag(d);
           beta1=(I./C+H1'*H1)\H1'*mytarget;
           K1=I./C+H1'*H1;
           %进入第一层隐含层在线学习步骤
           K1=K1+H1'*H1;
           beta1=beta1+K1\H1'*(mytarget-H1*beta1);
           H1=H1*beta1;
          [H1,SP]=mapminmax(H1,0,1);
           %初始化第二个隐含层节的参数
           a2=rand(qnum,N2);
           b2=rand(1,1);
           temH2=H1*a2+b2;
           %计算隐含层节点的输出值
           if strcmp(fun2,'sigmoid')==1
              H2=1./exp(-(temH2));
           elseif strcmp(fun2,'radbas')==1
              H2=radbas(temH2);
           elseif strcmp(fun2,'hardlim')==1
              H2=double(hardlim(temH2));
           elseif strcmp(fun2,'sine')==1
              H2=sin(temH2);
           end
           [H2,SP]=mapminmax(H2,0,1);
           %计算参数
           d=ones(1,N2);
           I=diag(d);
           beta2=(I./C+H2'*H2)\H2'*mytarget;
           K2=I./C+H2'*H2;
           %进入第二层隐含层在线学习步骤
           K2=K2+H2'*H2;
           beta2=beta2+K2\H2'*(mytarget-H2*beta2); 
        elseif (p+s>=pmin+3*smin)||(p>=t)%发生概念漂移
           %disp('conept drift');
           %更新隐含层神经元的数目
           N1=min(winsize,N1+floor(constant/(p+0.001)));
           N2=min(winsize,N2+floor(constant/(p+0.001)));
% %            N1=min(winsize,5*col+floor(constant/(p+0.001)));
% %            N2=min(winsize,floor(1.5*N1));
           %初始化第一个隐含层节点的参数取值
           a1=rand((col-1),N1);
           b1=rand(winsize,N1);
           temH=traindata*a1+b1;%计算矩阵的乘积
           %计算隐含层节点的输出值
           if strcmp(fun,'sigmoid')==1
              H1=1./exp(-(temH));
           elseif strcmp(fun,'radbas')==1
              H1=radbas(temH);
           elseif strcmp(fun,'hardlim')==1
              H1=double(hardlim(temH));
           elseif strcmp(fun,'sine')==1
              H1=sin(temH);
           end
           [H1,SP]=mapminmax(H1,0,1);
           %生成对角矩阵
           d=ones(1,N1);
           I=diag(d);
           beta1=(I./C+H1'*H1)\H1'*mytarget;
           K1=I./C+H1'*H1;
           H1=H1*beta1;
          [H1,SP]=mapminmax(H1,0,1);
           %初始化第二个隐含层节的参数
           a2=rand(qnum,N2);
           b2=rand(1,1);
           temH2=H1*a2+b2;
           %计算隐含层节点的输出值
           if strcmp(fun2,'sigmoid')==1
              H2=1./exp(-(temH2));
           elseif strcmp(fun2,'radbas')==1
              H2=radbas(temH2);
           elseif strcmp(fun2,'hardlim')==1
              H2=double(hardlim(temH2));
           elseif strcmp(fun2,'sine')==1
              H2=sin(temH2);
           end
           [H2,SP]=mapminmax(H2,0,1);
           %计算参数
           d=ones(1,N2);
           I=diag(d);
           beta2=(I./C+H2'*H2)\H2'*mytarget;
           K2=I./C+H2'*H2;
        end
        %开始进行测试
        temh1=testdata*a1+b1;
        %计算第一个隐含层节点的输出值
        if strcmp(fun,'sigmoid')==1
           h1=1./exp(-(temh1));
        elseif strcmp(fun,'radbas')==1
           h1=radbas(temh1);
        elseif strcmp(fun,'hardlim')==1
           h1=double(hardlim(temh1));
        elseif strcmp(fun,'sine')==1
           h1=sin(temh1);
        end
        h1=h1*beta1;%计算第二个隐含层的输入值
        [h1,SP]=mapminmax(h1,0,1);%标准化数据集
        temh2=h1*a2+b2;
        %计算第二个隐含层节点的输出值
        if strcmp(fun2,'sigmoid')==1
           h=1./exp(-(temh2));
        elseif strcmp(fun2,'radbas')==1
           h=radbas(temh2);
        elseif strcmp(fun2,'hardlim')==1
           h=double(hardlim(temh2));
        elseif strcmp(fun2,'sine')==1
           h=sin(temh2);
        end
        [h,SP]=mapminmax(h,0,1);
        output=h*beta2;%计算输出值
        [waste,result]=max(output,[],2);%获取真实的类标签
        p=1-countaccuracy(testtarget,result);
        accuracy=[accuracy,(1-p)];
        s=sqrt(p*(1-p)/c);
        %选取s和p的最小值
        if smin>s
            smin=s;
        end
        if pmin>p
            pmin=p;
        end
        num1=[num1,N1];
        num2=[num2,N2];
        disp(['第',num2str(c),'次测试的结果准确率为：',num2str(1-p),'  第一个隐含层节点数目为：',num2str(N1),'  第二个隐含层节点数目为：',num2str(N2)]);
        %toc;
    end
end
disp(['当前算法在此数据集上测试结果的准确率为',num2str(mean(accuracy)),'    第一个隐含层节点数目的平均值为：',num2str(mean(num1)),'    第二个隐含层节点数目的平均值为：',num2str(mean(num2))]);
disp(['算法在当前数据集上测试结果的准确率标准差为：',num2str(std(accuracy))]);
toc;
