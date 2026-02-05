## Benchmark with QE

Env
- QE: Intel(R) Xeon(R) Silver 4314 CPU @ 2.40GHz
- Jrystal: GPU: NVIDIA A100-SXM4-40GB (1 GPU)


### Diamond

#### Diamond (AE, primitive), LDA_X, K=1x1x1, 100 Bands, E_CUT=100 Eh

| Method  | Iterations | Energy (Eh) | JIT Time (s) | SCF Time (s) |
|---------|------------|-------------|--------------|--------------|
| QE      |   6        | -66.99195849|      -       | 10.72        |
| Jrystal |  13        | -66.99195892|    3.94      | 3.26         |


#### Diamond (AE, primitive), LDA_X, K=1x1x1, 100 Bands, E_CUT=200 Eh

| Method  | Iterations | Energy (Eh)  | JIT Time (s) | SCF Time (s) |
|---------|------------|--------------|--------------|--------------|
| QE      |   6        | -70.39413449 |      -       | 31.01        |
| Jrystal |   17       | -70.39413110 |    4.42      | 8.37         |


#### Diamond (AE, primitive), LDA_X, K=1x1x1, G=80x80x80, 100 Bands, E_CUT=400 Eh

| Method  | Iterations | Energy (Eh)  | JIT Time (s) | SCF Time (s) |
|---------|------------|--------------|--------------|--------------|
| QE      |   6        | -72.24084413 |      -       | 104.69       |
| Jrystal |   24       | -72.24084475 |    5.06      |  13.38       |


### Silicon

#### Silicon (AE, conventional), LDA_X, K=1x1x1, G=72x72x72, 100 Bands, E_CUT=50 Eh

| Method  | Iterations | Energy (Eh)  | JIT Time (s) | SCF Time (s) |
|---------|------------|--------------|--------------|--------------|
|QE (1 Core)  |   10   | -1476.73177625 |      -     |  98.38       |
|QE (16 Cores)|   10   | -1476.73177625 |      -     |  95.02       |
| Jrystal |   22       | -1476.73182702 |  5.73      |  13.83       |


#### Silicon (AE, conventional), LDA_X, K=1x1x1, G=96x96x96, 200 Bands, E_CUT=100 Eh

| Method  | Iterations | Energy (Eh)  | JIT Time (s) | SCF Time (s) |
|---------|------------|--------------|--------------|--------------|
|QE (1 Core)  |   10   | -1708.74550716 |      -     | 784.37       |
|QE (16 Cores)|   10   | -1708.74550716 |      -     | 182.33       |
|QE (AMD 96 Cores) | 9 | -1708.74550716 | -     | 681.15 |
| Jrystal |       32   | -1708.74553528 |  9.13      |  53.03       |
