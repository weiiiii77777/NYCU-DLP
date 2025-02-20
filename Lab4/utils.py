import numpy as np
import matplotlib.pyplot as plt

idx = []
for i in range(1, 101) : idx.append(i)

idx2 = []
for i in range(1, 201) : idx2.append(i)

idx3 = []
for i in range(1, 301) : idx3.append(i)

idx5 = []
for i in range(1, 501) : idx5.append(i)

def make(n_iter, n_cycle) :
    start = 0.0
    stop = 1.0
    ratio = 0.5
    L = np.ones(n_iter) * stop
    period = n_iter/n_cycle
    if ratio == 1 :
        ratio = (period - 1)/float(period)
    step = (stop-start)/(period*ratio)
    for c in range(n_cycle):
        v, i = start, 0
        while v <= stop and (int(i+c*period) < n_iter):
            L[int(i+c*period)] = v
            v += step
            i += 1
    return L

if __name__ == '__main__':

    epoch1 = make(100, 5)
    epoch2 = make(200, 10)
    epoch31 = make(300, 15)
    epoch32 = make(300, 10)
    epoch5 = make(500, 10)

    set1 = make(60, 3)
    set2 = make(60, 2)
    set3 = make(500, 10)

    plt.figure(figsize=(10,5))
    plt.plot(idx[:60], set1, label = '1')   
    plt.plot(idx[:60], set2, label = '2')
    plt.plot(idx[:60], set3[:60], label = '3')
    plt.xlabel('Epoch')    
    plt.ylabel('beta')
    plt.legend(['100 Epoch\n200 Epoch\n300 Epoch (M=15)', '300 Epoch (M=10)', '500 Epoch'], bbox_to_anchor=(1.0, 1.0), loc="upper left")
    plt.title('Different KL-term beta Curve', loc='center')
    plt.tight_layout()
    # plt.title('Optimizer = Adam', loc='right')
    plt.savefig('./curve.jpg')


# n_iter = 100
# n = 5
# r = 10 % n

# tfr_sde = 10
# min = 0.0
# tfr = 1
# tfr_d_step = 0.1
# result = []
# current_epoch = 0
# for i in range(n_iter):
#     if current_epoch >= tfr_sde and tfr > min and current_epoch % n == r:
#         tfr -= tfr_d_step
#     tfr = round(tfr, 2)
#     result.append(tfr)
#     current_epoch += 1
# print(result)


# 4, 5, 6, 10, 11 teacher forcing rate :
# [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 
#  0.9, 0.9, 0.9, 0.9, 0.9, 0.8, 0.8, 0.8, 0.8, 0.8, 
#  0.7, 0.7, 0.7, 0.7, 0.7, 0.6, 0.6, 0.6, 0.6, 0.6, 
#  0.5, 0.5, 0.5, 0.5, 0.5, 0.4, 0.4, 0.4, 0.4, 0.4, 
#  0.3, 0.3, 0.3, 0.3, 0.3, 0.2, 0.2, 0.2, 0.2, 0.2, 
#  0.1, 0.1, 0.1, 0.1, 0.1, 0.0, 0.0, 0.0, 0.0, 0.0, 
#  0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 
#  0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 
#  0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 
#  0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]

# 7, 8, 9 teacher forcing rate :
# [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 
#  0.9, 0.9, 0.9, 0.9, 0.9, 0.9, 0.9, 0.9, 0.9, 0.9, 
#  0.8, 0.8, 0.8, 0.8, 0.8, 0.8, 0.8, 0.8, 0.8, 0.8, 
#  0.7, 0.7, 0.7, 0.7, 0.7, 0.7, 0.7, 0.7, 0.7, 0.7, 
#  0.6, 0.6, 0.6, 0.6, 0.6, 0.6, 0.6, 0.6, 0.6, 0.6, 
#  0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 
#  0.4, 0.4, 0.4, 0.4, 0.4, 0.4, 0.4, 0.4, 0.4, 0.4, 
#  0.3, 0.3, 0.3, 0.3, 0.3, 0.3, 0.3, 0.3, 0.3, 0.3, 
#  0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 
#  0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1]

# cyclical kl annealing :
# [0.  0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9 
#  1.  1.  1.  1.  1.  1.  1.  1.  1.  1.  
#  0.  0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9 
#  1.  1.  1.  1.  1.  1.  1.  1.  1.  1.  
#  0.  0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9 
#  1.  1.  1.  1.  1.  1.  1.  1.  1.  1.  
#  0.  0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9 
#  1.  1.  1.  1.  1.  1.  1.  1.  1.  1.  
#  0.  0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9
#  1.  1.  1.  1.  1.  1.  1.  1.  1.  1. ]

# mono kl annealing :
# [0.   0.05 0.1  0.15 0.2  0.25 0.3  0.35 0.4  0.45 
#  0.5  0.55 0.6  0.65 0.7  0.75 0.8  0.85 0.9  0.95 
#  1.   1.   1.   1.   1.   1.   1.   1.   1.   1.   
#  1.   1.   1.   1.   1.   1.   1.   1.   1.   1.   
#  1.   1.   1.   1.   1.   1.   1.   1.   1.   1.   
#  1.   1.   1.   1.   1.   1.   1.   1.   1.   1.   
#  1.   1.   1.   1.   1.   1.   1.   1.   1.   1.
#  1.   1.   1.   1.   1.   1.   1.   1.   1.   1.   
#  1.   1.   1.   1.   1.   1.   1.   1.   1.   1.   
#  1.   1.   1.   1.   1.   1.   1.   1.   1.   1.  ]