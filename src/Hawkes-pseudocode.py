'''
Hawkes self-exciting
(concept, not real code)
'''

# 强度状态（以桶为例）
for i in range(B):   # B 个距离桶
    lamL[i] = muL[i]   # 当前强度
lamM = muM

t = 0
while t < T_end:
    # 1) 计算上界（指数核可用当前强度本身作为上界）
    upper = sum(lamL) + lamM + cancel_rate(X, theta)
    dt = np.random.exponential(1.0/upper)
    # 2) 强度随时间衰减（指数核 O(1) 更新）
    decay = np.exp(-beta*dt)          # 若各核不同，为每条强度用各自 beta
    for i in range(B):
        lamL[i] = muL[i] + (lamL[i]-muL[i]) * decay
    lamM  = muM  + (lamM - muM) * decayM
    t += dt

    # 3) 选择候选事件类型
    r = np.random.rand() * upper
    if r < sum(lamL):               # 候选=某桶的限价到达
        # 多项抽一个桶 j
        # thinning: 接受概率 = lamL[j] / (当前对该桶设置的上界)
        if np.random.rand() < 1.0:  # 上界取 lamL[j] 本身 → 恒接受
            place_limit_order(bucket=j, X)  # 更新 LOB
            lamL[j] += alphaLL_self          # 自激跳升
    elif r < sum(lamL) + lamM:      # 候选=市价买
        if np.random.rand() < 1.0:
            execute_at_best_ask(X)
            # 市价→回补自激：对近邻桶提升强度
            lamL[0] += alphaLM
            lamL[1] += alphaLM*np.exp(-c*1)
    else:                           # 候选=取消（保持原指数寿命机制）
        i = sample_bucket_proportional_to_volume(X)  # 按体量加权
        cancel_one_at_bucket(i, X)
