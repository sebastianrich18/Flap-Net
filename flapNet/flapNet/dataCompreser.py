
# removes 94% of unimportant data sets from data file

import numpy as np
import random as r

arr = np.loadtxt("data.txt", delimiter=',')
count = 0
for i in range(int(arr.shape[0]*.94)):
    index = r.randint(0,arr.shape[0]-1)
    if arr[index,3] == 0:
        arr = np.delete(arr,index,0)
        count += 1
        
if arr.shape[0] % 2 != 0:
    arr=np.delete(arr,1,axis=0)
    print('made even')
    
    
print('deleted ',count,'items')
np.savetxt('data.txt', arr, fmt='%1.3f', delimiter=',')
