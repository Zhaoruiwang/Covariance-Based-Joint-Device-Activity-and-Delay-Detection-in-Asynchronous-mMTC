function [y, t] = Ideal_CD(A, sampCov, sigma2, actset)
[L, N] = size(A);

gamma = zeros(N,1);
invSig = eye(L) / sigma2;

A_act = A(:, actset);
gamma_act = gamma(actset);

r = length(actset);

epsilon = 1e-3;


t = 0;
k = 0;

while k < 1000
    k = k + 1;
    
    invSigmaA = invSig * A_act;
    B = sampCov * invSigmaA;
    C = conj(invSigmaA) .* (A_act - B);
    g_act = real(sum(C))';
    
    
    % g = grad(gamma, A, sampCov, sigma2);
    
    if norm(max(gamma_act - g_act, 0) - gamma_act) < epsilon
        break
    end
    
    
    tic;
    idx_set = randperm(r);
    
    for iter = 1:r
        idx = idx_set(iter);
        % idx = unidrnd(N);
        a_vec = A_act(:,idx);
        
        %==============ML=====
        ainvS = a_vec'*invSig;
        b = ainvS*a_vec;
        b = real(b);
        e = real(ainvS*sampCov*ainvS');
        c = (e - b) / b^2;
        % c = real(ainvS*sampCov*ainvS' - b) / abs(b).^2;
        d = max(c, -gamma_act(idx));
        %=============Update===============
        if d ~= 0
            gamma_act(idx) = gamma_act(idx) + d;
            g = real(d*b);
            invSig = invSig - d/(1+g) * (ainvS'*ainvS);
        end
    end
    
    
    
    t = t + toc;
end


gamma(actset) = gamma_act;
y = gamma;
end