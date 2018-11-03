# Diffusion Simulation
## Assumption
1. A 3D space created with size of 64\*64\*64
2. There is not any other object in the space, as the polution reachs to the edge of space, it will rebound
3. The source of polution can be a block or a point. By default, it is a point(value=20000) in the position [x:32]-[y:32]-[z:32].
4. Let dt=1 , dx=1 , dy=1, dz=1, so that U(t+1) = (U(t)[x+1][y][z]+U(t)[x-1][y][z]+U(t)[x][y+1][z]+U(t)[x][y-1][z]+U(t)[x][y][z+1]+U(t)[x][y][z-1])/6

## Stretage
1. Create two kernal functions: `diffusion` and `refesh`. `Diffusion` is used to cacluate U(t+1) in term of U(t). `Refesh` is used to update the value in the space after cacluation step. 
2. There is a time loop in the host to call `diffusion` and `refesh` in order to achieve globle blocks synchronization.
3. A CPU version `diffusion` function created to validate the result and speed up.
4. Grid size = 64\*64\*64 ; Block size = 8\*8\*8

## Result
Since 3D object is hard to observe from outside, I choose the central slice which Z=32 to visualize the result when dt equals to 100,500,1000,3000,5000,10000 respectly. The surface plot can be found blew:

![img](img/visualDS.png)

`Figure[1-1]: srouce point = [x:32][y:32][z:32], slice Z=32`

As time goes by, the pollution starts to fill the whole space. Also the concetration of each postion trends to even. When dt=5000 or 10000, the shape of concetration is not a symmetrical structure anymore although the difference between maximum and minimun is very tiny. I think the possible reason is the start point is not the central of space, it is faster to reach one side than anther.

Then I changed the source of pollustion to [x:0][y:0][z:0], keep slice Z=32. The surface plot can be found blew:
![img](img/visualDS2.png)

`Figure[1-2]: srouce point = [x:0][y:0][z:0], slice Z=32`

The last two plots are almost as same as Figure[1-1].

## Speed up
Compared with CPU version, the GPU version running signifately faster which can reach up to 34x ~ 39x acclatation.

![img](img/Screenshot_20181103_172839.png)


| time/speed up | CPU      | GPU    | Speedup |
|---------------|----------|--------|---------|
| 100           | 362818   | 10536  | 34.44   |
| 500           | 1783376  | 51678  | 34.51   |
| 1000          | 3584415  | 107517 | 33.34   |
| 3000          | 10716670 | 291210 | 36.80   |
| 5000          | 17892344 | 452127 | 39.57   |
| 10000         | 35695658 | 925076 | 38.59   |
