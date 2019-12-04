# 计算H-hop的概率转移矩阵
def compute_diffusion(A,H):
    A_diffusion = [np.identity(A.shape[0])]

    if H > 0:
        A_sum = A.sum(0)
        A_stand = A/(A_sum+1.0)
        A_diffusion.append(A_stand)# +1.0平滑

        for i in range(2,H+1):
            A_diffusion.append(np.dot(A_stand, A_diffusion[-1]))
    return np.transpose(np.asarray(A_diffusion, dtype='float32'), (1,0,2))