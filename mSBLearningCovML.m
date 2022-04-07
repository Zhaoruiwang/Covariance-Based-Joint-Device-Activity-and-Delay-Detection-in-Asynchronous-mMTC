function [gamma_vec, MeanM, mse,mse_g,LF,LF_true,errStep] = mSBLearningCovML(A,Y,X,sigma2,N_iter,gamma_act)
% User Detection via ML from mSBL
% 2018-5-7 CZL 
M = size(Y,2);
[L, N] = size(A);
gamma_vec = ones(N,1)*max(gamma_act);
mse = zeros(N_iter,1); % OF THE SIGNAL X
mse_g = zeros(N_iter,1);
LF = zeros(N_iter,1);
LF_true = 0;
% LF_true = loglikelihoodFunction(A,Y,sigma2,gamma_act);

errStep = 0;
for iter = 1: N_iter
%     if mod(iter,10) == 1
%         display(strcat('iter=',num2str(iter)));
%     end
    
    Gamma_mtx = diag(gamma_vec);
    AGamma = A*Gamma_mtx;
    
    Sigma_t = sigma2*eye(L) + AGamma*A';
    invSig_t = pinv(Sigma_t);
    
    GammaAHinvSig_t = AGamma'*invSig_t;
    
    Sigma = Gamma_mtx - GammaAHinvSig_t*AGamma;
    MeanM = GammaAHinvSig_t*Y;
    
%     gamma_vec_neo = sum(abs(MeanM).^2,2)/M + real(diag(Sigma));
    gamma_vec_neo = sum(abs(MeanM).^2,2)/M./(1 - 1./gamma_vec.*real(diag(Sigma)));
    
    if max(gamma_vec_neo) == inf %% Sometimes err occurs
        display(strcat('!!! Inf appears at iter=',num2str(iter)));
        errStep = iter;
        mse(iter:end) = mse(iter-1);
        break;
    end
    
    
    
    gamma_vec = gamma_vec_neo;
    mse_iter = mean((gamma_act - gamma_vec).^2);
    mse_g(iter) = mse_iter;
    
%     mse_iter = mean(mean(abs(X - MeanM).^2));
%     mse(iter) = mse_iter;
%     
%     LF(iter) = loglikelihoodFunction(A,Y,sigma2,gamma_vec);
end

