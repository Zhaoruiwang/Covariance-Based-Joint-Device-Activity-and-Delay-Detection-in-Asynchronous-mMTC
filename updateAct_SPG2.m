function y = updateAct_SPG2(gamma_act, A_act, sampCov, sigma2, k_iter, epsilon_main)
%{
This function updates gamma in the selected support set
I import the number of iterations in order to use a method to slowly improve the accuracy
When support set selection is not good, we do not have to find exact solutions
%}
[~,r] = size(A_act);

M = 5; ratio = 2; c1 = 1e-2;
% These parameters are used for non-monotone line search

maxit = min(50, 10 * k_iter);
% maximum number of iterations on the support set

alpha_Min = 1e-10; alpha_Max = 1e10;  %%BB lower and upper bound
form = 1; % use for bb step

u = inf;
% upper bound of gamma ( 0 <= gamma <= u )

epsilon = max(0.8*epsilon_main, 1/10^k_iter);
% termination criterion
% I want this parameter to be well affected by k_iter,
% but it's not the best strategy right now


[valuef, Sigma] = f(gamma_act, A_act, sampCov, sigma2);

valueList = [valuef];

f_opt = valuef;
gamma_opt = gamma_act;

% ******************** starting iteration ********************

% Gradient in the support set

k = 0;
while k < maxit
    k = k + 1;
    
    % LL = chol(0.5*(Sigma+Sigma'));
    % invSigmaA = LL \ (LL' \ A_act);
    invSigmaA = Sigma \ A_act;
    B = sampCov * invSigmaA;
    C = conj(invSigmaA) .* (A_act - B);
    g_act = real(sum(C))';
    % Calculate the gradient at gamma_new
    
    
    if norm( min(max(gamma_act - g_act, 0), u) - gamma_act) < epsilon
        break
    end
    
    if k == 1
        alpha = 1;
    else
        g_new = g_act;
        y = g_new - g_old;
        
        sty = s' * y;
        if sty <= 0
            alpha = 1; form = 1;
        else
            if form == 1
                alpha = (s' * s) / sty; form = 2;
            else
                alpha = sty / (y' * y); form = 1;
            end % Alternate bb step
        end
    end
    
    alpha = min(max(alpha, alpha_Min), alpha_Max);
    
    gamma_old = gamma_act;
    gamma_new = gamma_old - alpha * g_act;
    
    gamma_new = min(max(gamma_new, 0), u); % projection
    
    direction = gamma_new - gamma_old;
    
    decrease_value = c1 * direction' * g_act;
    
    lambda = 1;
    s = direction;
    
    f_max = max(valueList);
    
    % ******************** Start non-monotone line search ********************
    
    while 1
        
        gamma_new = gamma_old + lambda * direction;
        
        [f_new, Sigma] = f(gamma_new, A_act, sampCov, sigma2);
        
        if lambda <= 1e-8
            break
        end
        
        if f_new <= f_max + lambda * decrease_value
            break
        else
            lambda = lambda / ratio;
            s = lambda * direction;
            gamma_new = gamma_old + s;
        end
        
        
    end
    
    % ******************** End non-monotone line search ********************
    
    g_old = g_act;
    gamma_act = gamma_new;
    
    valuef = f_new;
    
    valueList = [valueList, valuef];
    % Adds the function value to the valueList
    if length(valueList) > M
        valueList = valueList(2 : M + 1);
    end
    
    if valuef <= f_opt
        f_opt = valuef;
        gamma_opt = gamma_act;
    end
    
end % end while

fprintf('Iterate %d times within the active set, |active set| = %d\n', k, r);
y = gamma_opt;
end % end function