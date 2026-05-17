function sample = MH_Sampling(f_ks,sigma,xi,sample_min,sample_max)
%% Initialize the Metropolis sampler
T = 950000; % Set the maximum number of iterations
% Set sigma standard deviation of normal proposal density
% define a range for starting values
% 先单独尝试对每个变量生成MCMC
sample = zeros(T,1);
seed = 1; 
rand( 'state' , seed ); % set the random seed
randn( 'state' , seed );
% generate start value
sample(1) = unifrnd( sample_min , sample_max );
%% Start sampling
t = 1; 
while t < T % Iterate until we have T samples
	t = t + 1;  
	% 从任意一个概率q(x)采样，这里取正态
    sample_star = normrnd( sample(t-1) ,sigma );

	% 找到建议值在概率密度估计中的index
    index_sample_star = find(abs(sample_star - xi) == min(abs(sample_star  - xi)));
    
    % 找到历史值在概率密度估计中的index
    index = find(abs(sample(t-1,1) - xi) == min(abs(sample(t-1,1) - xi)));
    
    % 计算接受率
    alpha = min( [ 1 f_ks( index_sample_star ) / f_ks( index ) ] );
	
    % 判断是否接受
	u = rand;     
    if (u < alpha) && (sample_star >= sample_min) && (sample_star <= sample_max) 
        sample(t) = sample_star;
    else
        sample(t) = sample(t-1); % If not, copy old state
    end
end
end
