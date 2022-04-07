function A = sensingMatrixDesign(N,Q_max,L,type)
% Sequences
Numse_peruser=Q_max+1; % numer of sequence per user
Ne = N*Numse_peruser;
if strcmp(type, 'Gaussian')
    A_nodelay= (randn(L,N) + 1i*randn(L,N))*sqrt(0.5);
    A_delay=zeros(L+Q_max,Ne);
    for n=1:N
        for d=1:Q_max+1
            A_delay(d:d+L-1,(n-1)*Numse_peruser+d)=A_nodelay(:,n);
        end
    end
    A=A_delay;  % generate psedo sensing matrix   
elseif strcmp(type, 'QAM')
    A = randi(4,L,Ne);
    A(A == 1) = 1+1j;
    A(A == 2) = 1-1j;
    A(A == 3) = -1+1j;
    A(A == 4) = -1-1j;
    A = A*sqrt(0.5);
    
else
    error('oooooops matrix');
end