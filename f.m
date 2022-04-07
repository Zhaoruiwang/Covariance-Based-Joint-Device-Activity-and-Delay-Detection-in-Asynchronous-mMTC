function [f_value, Sigma] = f(gamma, A, sampCov, sigma2)
% Calculate the value of the objective function
[L, ~] = size(A);
%{
supp = find(gamma);
gamma_supp = gamma(supp);
A_supp = A(:, supp);
Sigma = (A_supp .* gamma_supp') * A_supp' + sigma2 * eye(L);
%}
Sigma = (A .* gamma') * A' + sigma2 * eye(L);

LL = chol(0.5*(Sigma+Sigma'));

f_log = 2 * sum(log(diag(LL)));

f_value = f_log + trace(LL \ (LL' \ sampCov));
f_value = real(f_value);
end