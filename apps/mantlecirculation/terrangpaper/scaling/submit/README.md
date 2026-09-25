# Sweep generators and collector

Generate sweep job scripts for machines other than SuperMUC-NG and LUMI, and
collect the results.

| file | purpose |
|---|---|
| `submit_bench_mt_lumi.py` | generate a sweep for LUMI-G |
| `submit_bench_mt_mn5.py`  | generate a sweep for MareNostrum 5 |
| `submit_bench_mt_jb.py`   | generate a sweep for JUWELS Booster |
| `submit_mc_lumi.py`       | generate mantle-circulation production jobs for LUMI-G |
| `collect_bench_mt.py`     | read the emitted manifest and write a results CSV |

Set `TERRANG_BIN` and `TERRANG_ROOT` before running. The account is a
`{ACCOUNT}` template field filled in at generation time.

`../sng2_reproduction` and `../lumi_reproduction` are the two sweeps actually
reproduced here and are the better starting point if your machine resembles
either.
