function H = channelGeneration(N,M,sigma2s)
% Generate channel

H = sqrt(1/2)*(randn(N,M) + 1i*randn(N,M));
H = repmat(sigma2s,1,M).^0.5.*H; % sigma2s large-scale fading component