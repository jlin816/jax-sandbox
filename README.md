## Torch, GPU:
```
step 0 | elapsed 0.14m | loss: 4.12 | val: 3.53
step 100 | elapsed 0.90m | loss: 3.16 | val: 2.97
step 200 | elapsed 1.67m | loss: 3.25 | val: 2.93
step 300 | elapsed 2.43m | loss: 3.21 | val: 2.91
step 400 | elapsed 3.20m | loss: 3.02 | val: 2.88
step 500 | elapsed 3.97m | loss: 3.21 | val: 2.87
```

## Jax, GPU, no jit:
(TODO: these #s don't include eval time)
```
step 0 | elapsed 0.25m | loss: 4.01
step 100 | elapsed 2.10m | loss: 3.87
step 200 | elapsed 3.95m | loss: 3.88
step 300 | elapsed 5.84m | loss: 3.98
step 400 | elapsed 7.68m | loss: 3.91
step 500 | elapsed 9.52m | loss: 3.92
```

## Jax, GPU, jit train step:
(Note first step is slower than torch because of jit!) 
```
step 0 | elapsed 0.38m | loss: 4.01
step 100 | elapsed 0.56m | loss: 3.87
step 200 | elapsed 0.75m | loss: 3.88
step 300 | elapsed 0.94m | loss: 3.98
step 400 | elapsed 1.12m | loss: 3.91
step 500 | elapsed 1.31m | loss: 3.92
```
