function [gamma_vec, CDinfo,mse_iter,fval_iter] = coordDescentCovML_fast_tol_mse(A,sampCov,sigma2,g_act,tol1,maxN_itera,Y)
% User Detection via ML from covariance matrix
% 2018-3-9 CZL
% 2018-5-27
tic;


debug_mode = 'off';
[L, N] = size(A);
[~, M] = size(Y);
% mse = zeros(N_iter,1);

Sigma = eye(L)*sigma2;
invSig = eye(L)./sigma2;
gamma_vec = zeros(N,1);
% mse_iter = mean((g_act).^2);

% mse = [];
% mse_sub = zeros(N,1);
%%===============================
mse_iter = [];
fval_iter = [];
itt = 1;
cnt = 0;
if strcmp(debug_mode,'on')
    mse_iter(itt) = mean((g_act - gamma_vec).^2);
    fval_iter(itt) = real(logdet(Sigma) + trace(Sigma\sampCov));
    itt = itt + 1;
end


while 1
    cnt = cnt + 1;
   
    %{
    invSigmaA = Sigma\A;
    hatSigmainvSigmaA = sampCov * invSigmaA;
    Temp = conj(invSigmaA) .* (A - hatSigmainvSigmaA);
    gradient = real(sum(Temp))';
    % compute the whole gradient
    
    % check the stopping criterion
    projgamma = max(gamma_vec - gradient, 0);
    residual = norm(gamma_vec - projgamma);
    %     if mod(cnt,10) == 0
    %         disp(cnt);
    %     end
    %    [vall,idxx] = max(abs(gamma_vec - projgamma));
    %     disp([residual,vall,idxx]);
    %     disp(sqrt(sigma2s_eff(idxx)));
    
    if residual <= tol1 || cnt > maxN_itera
        CDinfo.fval = real( log(det(Sigma)) + trace(Sigma\sampCov) );
        CDinfo.projerr = residual;
        CDinfo.xerr = norm(gamma_vec - g_act);
        CDinfo.flag = 1;
        CDinfo.sparsity = nnz(gamma_vec);
        return;
    end
    %}
   
     if cnt > maxN_itera
        break;
    end
    
    idx_set = randperm(N);
    
    for iter = 1:N
        %     idx = randi(N);
        idx = idx_set(iter);
%         idx = UptSet(iter);
        a_vec = A(:,idx);
        
        %==============ML=====
        ainvS = a_vec'*invSig;
        b = ainvS*a_vec;
        c = real(ainvS*sampCov*ainvS' - b)/abs(b).^2;
        
        d = max(c,-gamma_vec(idx));
        %=============Update===============
        if d ~= 0
            gamma_vec(idx) = gamma_vec(idx) + d;
            a_mtx = d*(a_vec*a_vec');
            Sigma = Sigma + a_mtx;
            
            g = real(d*b);
            invSig_neo  = invSig - d/(1+g)*(ainvS'*ainvS);
            invSig = invSig_neo;
            
        end
      
    end
   
    if strcmp(debug_mode,'on')
        mse_iter(itt) = mean((g_act - gamma_vec).^2);
        fval_iter(itt) = real(logdet(Sigma) + trace(invSig*sampCov));
        itt = itt + 1;
        
    end
end

ttt = toc;
CDinfo.ttt = ttt;
realCov = A*diag(g_act)*A'+eye(L)*sigma2;
CDinfo.truef = real(logdet(realCov) + trace(realCov\sampCov));

end

function y = logdet(X)
LL = chol(0.5*(X+X'));
y = 2*sum(log(diag(LL)));
end

