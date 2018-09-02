import numpy as np
import time

v_k,v_n = 50,250000

def pickNum(k,n):
    np.random.seed(500)
    serise = np.arange(1,n+1)
    output = sorted(np.random.choice(serise,k,replace=False))
    return output

start = time.time()
print('Output:',pickNum(v_k,v_n))
end = time.time()
print('Spend:',str(end-start)+'s')
