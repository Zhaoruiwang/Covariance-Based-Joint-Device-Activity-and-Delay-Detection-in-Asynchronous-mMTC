function [x,user_supp,supp,user_idx,delay_idx,sigma2n] = signalGeneration(N,K,L,Q_max,M,H,txPowerMax,noisePower)
% Generate the activity pattern and the signal normalized
% 2019-07-15
Q=Q_max+1; % number of sequence per user
% User activity pattern
user_idx = randperm(N);
user_idx(1:K) = sort(user_idx(1:K)); % user_supp(1:K), the index of active users
user_supp = zeros(N,1);
user_supp(user_idx(1:K)) = 1;

% Delay of active users 
delay_idx = randi(Q,K,1); % the data indices for active users

% Combined support
supp = zeros(Q*N,1);   
supp(sort(user_idx(1:K)-1)'*Q + delay_idx) = 1;

% Signal generation 
Ne = N*Q;
Heff = repelem(H,Q,1); % eff. channel; channel for sequences of one users are equal.
x = zeros(Ne,M);
x((supp==1),:) = Heff((supp==1),:); 


% Noise setup with power control
sigma2n = (10^((noisePower)/10)); % 
txPower = 10^(txPowerMax/10);
sigma2n = sigma2n/txPower;

