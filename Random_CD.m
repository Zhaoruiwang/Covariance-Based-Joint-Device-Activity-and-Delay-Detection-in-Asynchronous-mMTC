function [y, actset_es,cov_time] = Random_CD(A, sampCov, sigma2, Q_max,thd)

%{
Coordinate descent algorithm
The termination condition is || [x - \nabla f(x)]_+ - x || < epsilon
%}
[L, N] = size(A);

gamma = zeros(N,1);
invSig = eye(L) / sigma2;


k = 0;

epsilon = 1e-3;
fvalue_now = f(gamma, A, sampCov, sigma2);
N_iter=200;
while k < N_iter
    k = k + 1;
    fvalue_last=fvalue_now;
    
    idx_set = randperm(N);
    
    for iter = 1:N
        idx = idx_set(iter);
        % idx = unidrnd(N);
        a_vec = A(:,idx);
        
        %==============ML=====
        ainvS = a_vec'*invSig;
        b = ainvS*a_vec;
        b = real(b);
        e = real(ainvS*sampCov*ainvS');
        c = (e - b) / b^2;
        % c = real(ainvS*sampCov*ainvS' - b) / abs(b).^2;
        d = max(c, -gamma(idx));
        %=============Update===============
        if d ~= 0
            gamma(idx) = gamma(idx) + d;
            g = real(d*b);
            invSig = invSig - d/(1+g) * (ainvS'*ainvS);
        end
    end
    fvalue_now=f(gamma, A, sampCov, sigma2);
    abs_value=abs(fvalue_last-fvalue_now);
    if abs_value < epsilon
        break
    end
end

cov_time=k;
Blk_lth=Q_max+1;
Blk_num=N/(Q_max+1);
for b=1:Blk_num
    gamma_b=gamma((b-1)*Blk_lth+1:b*Blk_lth);
    [gamma_b_max index]=max(gamma_b);
    if gamma_b_max>=thd
        set=(b-1)*Blk_lth+1:b*Blk_lth;
        set(index)=[];
        gamma(set)=0;
    else
        gamma((b-1)*Blk_lth+1:b*Blk_lth)=0;
    end
end

y = gamma;
actset_es=find(gamma);
end

