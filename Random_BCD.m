function [y, actset_es,cov_time] = Random_BCD(A, sampCov, sigma2, Q_max,thd)

%{
Coordinate descent algorithm
The termination condition is || [x - \nabla f(x)]_+ - x || < epsilon
%}

[L, N_all] = size(A);
blk_lgh=Q_max+1;
N=N_all/blk_lgh;
gamma = zeros(N_all,1);
invSig_n = eye(L) / sigma2;
fvalue_n = f(gamma, A, sampCov, sigma2); 
epsilon = 1e-3;

k = 0;


N_ter=200;

while k < N_ter
    k = k + 1;
%     tic
    
    fvalue_n_last=fvalue_n;
    idx_set = randperm(N);
    for iter = 1:N
        idx = idx_set(iter);
        gamma_n=gamma((idx-1)*blk_lgh+1:idx*blk_lgh); %  original gamma_n at the n_th block
        [gamma_nmax idx_max]=max(gamma_n);
        f_value=zeros(blk_lgh,1);
        Gamma_n=zeros(blk_lgh,blk_lgh); % all gamma at the n-th block
        INVSIG_n=[];
        Fvalue_n=[];
        InvSig_ntau=[];
        for tau=1:blk_lgh
            % compute covariance matrix
            gamma_ntau=gamma_n(tau);
            if gamma_nmax==0 || tau==idx_max
               invSig_ntau=invSig_n; 
               fvalue_ntau=fvalue_n;
            elseif tau~=idx_max
                a_vec = A(:,(idx-1)*blk_lgh+idx_max);
                ainvS = a_vec'*invSig_n;
                bb = real(ainvS*a_vec);
                gg = real(gamma_nmax*bb);
                ee = real(ainvS*sampCov*ainvS');
                invSig_ntau = invSig_n + gamma_nmax/(1-gg) * (ainvS'*ainvS);
                fvalue_ntau=fvalue_n+log(1-gg)+gamma_nmax*ee/(1-gg);
            end 
            a_vec = A(:,(idx-1)*blk_lgh+tau);
            ainvS = a_vec'*invSig_ntau;
            b = real(ainvS*a_vec);          
            e = real(ainvS*sampCov*ainvS');
            c = (e - b) / b^2;
            d = max(c, -gamma_ntau);
            if d ~= 0
               gamma_ntau = gamma_ntau + d;
               g = real(d*b);
               invSig_ntau = invSig_ntau - d/(1+g) * (ainvS'*ainvS);
               fvalue_ntau=fvalue_ntau+log(1+g)-d*e/(1+g);
            end 
            Gamma_n(tau,tau)=gamma_ntau;
            gamma((idx-1)*blk_lgh+1:idx*blk_lgh)=Gamma_n(tau,:);
            Fvalue_n=[Fvalue_n;fvalue_ntau];  
            INVSIG_n=[INVSIG_n;invSig_ntau];
        end
        [~,idx_mintau]=min(Fvalue_n);
        gamma((idx-1)*blk_lgh+1:idx*blk_lgh)=Gamma_n(idx_mintau,:);
        invSig_n=INVSIG_n((idx_mintau-1)*L+1:idx_mintau*L,:);
        fvalue_n=Fvalue_n(idx_mintau);
    end
    abs_value=abs(fvalue_n-fvalue_n_last);
    if abs_value < epsilon
        break
    end
%      toc
end
y = gamma;
cov_time=k;
actset_es=find(gamma>thd);
end

