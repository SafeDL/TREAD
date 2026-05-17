function sample = Multi_MH_Sampling(f_ks,xi)
%% Initialize the Metropolis sampler，效果不好
T = 200; % Set the maximum number of iterations
% Set sigma standard deviation of normal proposal density
% define a range for starting values
% 先单独尝试对每个变量生成MCMC
sample = zeros(T,3);
seed = 1; 
rand( 'state' , seed ); % set the random seed
randn( 'state' , seed );
% generate start value
sample(1,1) = 0;
sample(1,2) = 0;
sample(1,3) = 0;
%% Start sampling
t = 1; 
sigma = [1 0.03 0.04
   0.03 0.015 0.01
   0.04 0.01 0.02];

while t < T % Iterate until we have T samples
	t = t + 1;  
	% 从任意一个概率q(x)采样，这里取正态
    sample_star = sample(t-1,:) +  randn(1,3);
%     sample_star = mvnrnd( sample(t-1,:) ,sigma );

	% 找到建议值在概率密度估计中的index
    temp = sample_star - xi;
    index_sample_star = 1;
    for i = 1:length(temp)
        if norm(temp(i,:)) < norm(temp(index_sample_star,:))
            index_sample_star = i;
        else
            continue;
        end
    end
    
    % 找到历史值在概率密度估计中的index
    temp = sample(t-1,:) - xi;
    index = 1;
    for i = 1:length(temp)
        if norm(temp(i,:)) < norm(temp(index,:))
            index = i;
        else
            continue;
        end
    end
    
    % 计算接受率
    alpha = min( [ 1 f_ks( index_sample_star ) / f_ks( index ) ] );
	
    % 判断是否接受
	u = rand;     
    if (u < alpha)  
        sample(t,:) = sample_star;
    else
        sample(t,:) = sample(t-1,:); % If not, copy old state
    end
end
end
