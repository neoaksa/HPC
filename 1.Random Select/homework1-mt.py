# muliti threading
import numpy as np
import time
from multiprocessing import Pool
from functools import partial

v_k,v_n = 50,250000

def pickNum(sliceindex,thread,k,n):
    np.random.seed(sliceindex)
    slice_size = int(n/thread)
    start_num = (sliceindex-1)*slice_size + 1
    end_num = sliceindex * slice_size + 1
    serise = np.arange(start_num, end_num)
    output = sorted(np.random.choice(serise,k,replace=False))
    return output

start = time.time()
thread_num = 5 # thread number
p = Pool(thread_num)
parm = partial(pickNum,thread=thread_num,k=int(v_k/thread_num),n=v_n) 
result = np.array(p.map(parm,np.arange(1,thread_num+1)))
print('Output:')
print(result.flatten())
end = time.time()
print('Spend:',str(end-start)+'s')
