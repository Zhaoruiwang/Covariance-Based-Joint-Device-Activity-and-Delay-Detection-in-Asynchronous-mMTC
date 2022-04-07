function [act_set,act_set_es,cov_time] = test_asyn(N, K, L, Q_max, M,thd_CD,thd_BCD)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% In this function we perform an experiment on all algorithms
% Note that the power is normalized by 128dBm
% N = 1000;
% K = 150;
% L = 64;
% J = 2;
% M = 128;
% maxN_itera = 50;
txPower = 23; % dBm
noisePower = -99; % dBm

sigma2s = ones(N,1);  % large-scale fading
txPowerN = 0;
noisePowerN = noisePower + 128 - txPower;
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

matrx_Type = 'Gaussian';

% Sequence generation
A = sensingMatrixDesign(N,Q_max,L,matrx_Type);

% Gaussian channel
H = channelGeneration(N,M,sigma2s);

% Sparse signal
[x,user_supp,supp,user_idx,data_idx,sigma2n] = signalGeneration(N,K,L,Q_max,M,H,txPowerN,noisePowerN);
act_set=find(supp);
% Additive noise
w = sqrt(1/2)*(randn(L+Q_max,M)+1i*randn(L+Q_max,M))*sqrt(sigma2n);

% System model
y = A * x + w;
sampCov = (1/M)*(y*y'); sigma2 = sigma2n;

fprintf('CD\n');
  % [gamma_cd,  act_set_es,cov_time] = Random_CD(A, sampCov, sigma2, Q_max,thd_CD); % with prior information
   [gamma_bcd,  act_set_es,cov_time] = Random_BCD(A, sampCov, sigma2, Q_max,thd_BCD);


end