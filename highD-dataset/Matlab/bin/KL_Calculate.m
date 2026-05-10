function score_kl = KL_Calculate(p,q)
% 参数解释：p:原始概率密度向量
% q:MC采样后的概率密度向量
% score_k1：KL散度得分，刻画替代概率密度对原始概率密度的信息保留程度
if p == q
    score_kl = 0;
else
    score_kl = sum(p.* log(eps + p./(q+eps)));
end
end