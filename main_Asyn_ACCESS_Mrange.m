clear;
clc;

N = 200; % total users
K=120;

Q_max=3; % maximum delay symbols
L=104;
M_range = [10:10:50]; % The length of the sequence
%M_range=[4:2:12];
ite = 20000; % All experiments are repeated ite times
thd_CD=0.2; % for CD, thd=0.1 for all cases; for benchmark, K=50 thd=0.13; K=90 thd=0.17; K=120 thd=0.2.
if K==50
    thd_BCD=0.11; %for BCD, thd=0.15 for K=120; thd=0.12 for K=90; thd=0.11 for K=50;
elseif K==90
    thd_BCD=0.12;
elseif K==120
    thd_BCD=0.15;
end 
CDE=zeros(1,length(M_range));
ITE=zeros(1,length(M_range));
FAE=zeros(1,length(M_range));
SZ=zeros(1,length(M_range));% estimation size
for l = 1:length(M_range)
    M=M_range(l);
    for j = 1:ite
        j
        tic
        [act_set,act_set_es,cov_time] = test_asyn(N, K, L, Q_max, M,thd_CD,thd_BCD);
        corr_detc=intersect(act_set,act_set_es);
        a=size(act_set_es,1);
        b=size(corr_detc,1);
        fs_dec=setdiff(act_set_es,corr_detc);
        CDE(l) = CDE(l) + length(corr_detc)/K;
        FAE(l) = FAE(l) + length(fs_dec)/((Q_max+1)*N-K);
        ITE(l) = ITE(l) + cov_time;
        SZ(l) = SZ(l) + size(act_set_es,1);
        toc
    end
    CDE(l)=CDE(l)/ite;
    FAE(l)=FAE(l)/ite;
    ITE(l)=ITE(l)/ite;
    SZ(l)=SZ(l)/ite;
end
%%
MDE=1-CDE;
figure
subplot(211);
semilogy(M_range, MDE);
xlabel('M');
ylabel('Missed Detection Probability');
grid on;
hold on;

subplot(212);
semilogy(M_range, FAE);
xlabel('M');
ylabel('False Alarm Probability');
grid on;
hold on;

figure
plot(M_range,ITE)
xlabel('M');
ylabel('Iterate Time');
grid on

figure
plot(M_range,SZ)
xlabel('M');
ylabel('Iterate Time');
grid on
legend('CD: \tau_{{\rm max}}=4 and L=100','BCD: \tau_{{\rm max}}=4 and L=100','CSA: \tau_{{\rm max}}=4 and L=100','CD: \tau_{{\rm max}}=0 and L=104')