function sample = Slice_Sampling(fittedmodel,sample_min,sample_max)
% fittedmodel - 拟合的概率密度
seed = 1; 
rand( 'state' , seed ); % set the random seed
randn( 'state' , seed );
targetpdf = @(x) fittedmodel(x); 
nsamples = 20000;
x_init = unifrnd( sample_min , sample_max );
sample = slicesample(x_init,nsamples,'pdf',targetpdf,'burnin',1000);
end
