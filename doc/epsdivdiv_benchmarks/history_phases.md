| Phase | Versions | Endpoint | Gdof/s | Gain | AI (F/B) | GFLOP/s | regs | DRAM GB | spill MB |
|---|---|---|---|---|---|---|---|---|---|
| baseline | v00a | v00a | 0.012 | — | 4.77 | 1551 | 255 | 13565.29 | 293741403.3 |
| 1. arithmetic reduction | v00a-v02b | v02b | 0.112 | 9.2x | 0.09 | 124 | 255 | 6191.73 | 5927222.6 |
| 2. teams + shared memory | v03-v05 | v05 | 8.788 | 78.7x | 27.78 | 8022 | 108 | 16.60 | 165591.1 |
| 3. data layout & tiling | v06-v08 | v08 | 11.860 | 1.3x | 14.27 | 4913 | 118 | 14.67 | 197803.4 |
| 4. register / occupancy tuning | v09-v10 | v10 | 17.100 | 1.4x | 14.71 | 6879 | 96 | 13.82 | 29171.4 |

Cumulative v00a → v10: **1412×**
