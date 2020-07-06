tic;
data=krkopt;%���ݼ�
t=0.05;
winsize=1000;%�������ڵĴ�С
col=size(data,2);%���ݵ�����
%���ݽ��б�׼��
tempdata=zscore(data(:,1:(col-1)));
data=[tempdata,data(:,col)];
clear tempdata;%�����ʱ����
constant=5;
fun='sigmoid';
fun2='sigmoid';
accuracy=[];%����ÿһ�η�������׼ȷ��
%��ʼ�����
num1=[];%�����һ��������ڵ����Ŀ
num2=[];%����ڶ���������ڵ����Ŀ
N1=5*col;%��һ��������ڵ����Ŀ
N2=5*col;%�ڶ���������ڵ����Ŀ
C=0.003;
p=0;
pmin=1;
smin=1;
c=0;%��¼���Ե�
qnum=max(data(:,col));%���ǩ�����ֵ
s=sqrt(p*(1-p)/(c+1));
for i=1:size(data,1)
    if (i>=2*winsize)&&(mod(i,2*winsize)==0)
        traindata=data((i-2*winsize+1):(i-winsize),1:(col-1));%��ȡѵ����
        traintarget=data((i-2*winsize+1):(i-winsize),col);%��ȡѵ��Ŀ���������ǩ
        testdata=data((i-winsize+1):i,1:(col-1));%��ȡ���Լ�
        testtarget=data((i-winsize+1):i,col);%��ȡ�����������ǩ
        %��ѵ������Ŀ��ֵ����0-1��
        c=c+1;%��һ�β���
        mytarget=zeros(size(traindata,1),qnum);%mytargetΪѵ�����Ĳ������ݵ����ǩ0-1�����
        for pp=1:size(traintarget,1)
            mytarget(pp,traintarget(pp,1))=1;
        end
        if (p+s)<=(pmin+2*smin)&&(p<t)%˵����ǰ��������
           %��ʼ����һ��������ڵ�Ĳ���ȡֵ
           %disp('������');
           a1=rand((col-1),N1);
           b1=rand(1,1);
           temH=traindata*a1+b1;%�������ĳ˻�
           %����������ڵ�����ֵ
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
           %���ɶԽǾ���
           d=ones(1,N1);
           I=diag(d);
           beta1=(I./C+H1'*H1)\H1'*mytarget;
           K1=I./C+H1'*H1;
           %�����һ������������ѧϰ����
           K1=K1+H1'*H1;
           beta1=beta1+K1\H1'*(mytarget-H1*beta1);
           H1=H1*beta1;
           [H1,SP]=mapminmax(H1,0,1);
           %��ʼ���ڶ���������ڵĲ���
           %=zscore(H1);
           a2=rand(qnum,N2);
           b2=rand(1,1);
           temH2=H1*a2+b2;
           %����������ڵ�����ֵ
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
           %�������
           d=ones(1,N2);
           I=diag(d);
           beta2=(I./C+H2'*H2)\H2'*mytarget;
           K2=I./C+H2'*H2;
           %����ڶ�������������ѧϰ����
           K2=K2+H2'*H2;
           beta2=beta2+K2\H2'*(mytarget-H2*beta2);
        elseif (p+s>pmin+2*smin)&&(p+s<pmin+3*smin)%�ﵽ����ˮƽ
            %disp('���棡');
            %������������Ԫ����Ŀ
            N2=N1;
            N1=min(winsize,N1+floor(constant/(p+0.001)));
           %��ʼ����һ��������ڵ�Ĳ���ȡֵ
           a1=rand((col-1),N1);
           b1=rand(1,1);
           temH=traindata*a1+b1;%�������ĳ˻�
           %����������ڵ�����ֵ
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
           %���ɶԽǾ���
           d=ones(1,N1);
           I=diag(d);
           beta1=(I./C+H1'*H1)\H1'*mytarget;
           K1=I./C+H1'*H1;
           %�����һ������������ѧϰ����
           K1=K1+H1'*H1;
           beta1=beta1+K1\H1'*(mytarget-H1*beta1);
           H1=H1*beta1;
          [H1,SP]=mapminmax(H1,0,1);
           %��ʼ���ڶ���������ڵĲ���
           a2=rand(qnum,N2);
           b2=rand(1,1);
           temH2=H1*a2+b2;
           %����������ڵ�����ֵ
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
           %�������
           d=ones(1,N2);
           I=diag(d);
           beta2=(I./C+H2'*H2)\H2'*mytarget;
           K2=I./C+H2'*H2;
           %����ڶ�������������ѧϰ����
           K2=K2+H2'*H2;
           beta2=beta2+K2\H2'*(mytarget-H2*beta2); 
        elseif (p+s>=pmin+3*smin)||(p>=t)%��������Ư��
           %disp('conept drift');
           %������������Ԫ����Ŀ
           N1=min(winsize,N1+floor(constant/(p+0.001)));
           N2=min(winsize,N2+floor(constant/(p+0.001)));
% %            N1=min(winsize,5*col+floor(constant/(p+0.001)));
% %            N2=min(winsize,floor(1.5*N1));
           %��ʼ����һ��������ڵ�Ĳ���ȡֵ
           a1=rand((col-1),N1);
           b1=rand(winsize,N1);
           temH=traindata*a1+b1;%�������ĳ˻�
           %����������ڵ�����ֵ
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
           %���ɶԽǾ���
           d=ones(1,N1);
           I=diag(d);
           beta1=(I./C+H1'*H1)\H1'*mytarget;
           K1=I./C+H1'*H1;
           H1=H1*beta1;
          [H1,SP]=mapminmax(H1,0,1);
           %��ʼ���ڶ���������ڵĲ���
           a2=rand(qnum,N2);
           b2=rand(1,1);
           temH2=H1*a2+b2;
           %����������ڵ�����ֵ
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
           %�������
           d=ones(1,N2);
           I=diag(d);
           beta2=(I./C+H2'*H2)\H2'*mytarget;
           K2=I./C+H2'*H2;
        end
        %��ʼ���в���
        temh1=testdata*a1+b1;
        %�����һ��������ڵ�����ֵ
        if strcmp(fun,'sigmoid')==1
           h1=1./exp(-(temh1));
        elseif strcmp(fun,'radbas')==1
           h1=radbas(temh1);
        elseif strcmp(fun,'hardlim')==1
           h1=double(hardlim(temh1));
        elseif strcmp(fun,'sine')==1
           h1=sin(temh1);
        end
        h1=h1*beta1;%����ڶ��������������ֵ
        [h1,SP]=mapminmax(h1,0,1);%��׼�����ݼ�
        temh2=h1*a2+b2;
        %����ڶ���������ڵ�����ֵ
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
        output=h*beta2;%�������ֵ
        [waste,result]=max(output,[],2);%��ȡ��ʵ�����ǩ
        p=1-countaccuracy(testtarget,result);
        accuracy=[accuracy,(1-p)];
        s=sqrt(p*(1-p)/c);
        %ѡȡs��p����Сֵ
        if smin>s
            smin=s;
        end
        if pmin>p
            pmin=p;
        end
        num1=[num1,N1];
        num2=[num2,N2];
        disp(['��',num2str(c),'�β��ԵĽ��׼ȷ��Ϊ��',num2str(1-p),'  ��һ��������ڵ���ĿΪ��',num2str(N1),'  �ڶ���������ڵ���ĿΪ��',num2str(N2)]);
        %toc;
    end
end
disp(['��ǰ�㷨�ڴ����ݼ��ϲ��Խ����׼ȷ��Ϊ',num2str(mean(accuracy)),'    ��һ��������ڵ���Ŀ��ƽ��ֵΪ��',num2str(mean(num1)),'    �ڶ���������ڵ���Ŀ��ƽ��ֵΪ��',num2str(mean(num2))]);
disp(['�㷨�ڵ�ǰ���ݼ��ϲ��Խ����׼ȷ�ʱ�׼��Ϊ��',num2str(std(accuracy))]);
toc;
