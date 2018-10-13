### Reduction Sum
Cuda version(GTX950 128bit 6600Mhz):
`workperform.cc`: main function which invokes `innerproduct` in cuda file.
`workperform.cu`: cuda file, inculds reduction sum function.

CPU version(E3-1230 v3 3.3GHz)
`CPU.cc`: single thread running on cpu for evaluation

### Performance analysis
1. Speed up

![img](imgs/chart.png)

| Data Size | CPU(usec) | GPU(usec) | Speedup   | 
|-----------|-----------|-----------|-----------| 
| 2^10      | 7         | 80166     | 0.000087  | 
| 2^20      | 6334      | 79361     | 0.079813  | 
| 2^30      | 5654058   | 180244    | 31.368911 | 

When data size is relatively small, CPU is running faster than GPU, since there is a latency from memory to GPU. But when the data size grow up to 2^30, the GPU is more than 30X faster than single thread on CPU. 

2. Bandwidth Utility

Actual Bandwidth = 2^30\*4/1000/1000/180.244=23.82GB/s

Max Bandwidth=105.6GB/s

E = 23.82/105.6=23%
