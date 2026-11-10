# SuperMUC-NG Phase 2 strong-scaling sweep

Produces the SuperMUC-NG line of Fig. 5 of the preprint.

34 standard-mode points, MT32 on 1 device to MT1024 on 512 devices, 8 ranks per
node on Intel PVC. Named `run_MT<level>_g<devices>_A3_oe.sh`.

## Running

```
TERRANG_BIN=<path-to>/mantlecirculation \
  sbatch --account=<your-account> run_MT256_g64_A3_oe.sh
```

`TERRANG_CFG` defaults to `../config_scal_A3.toml`, `TERRANG_OUT` to a directory
named after the point. The Slurm log lands next to the script.

Per-step time comes from `<out>/timer_trees/timer_tree_9.json`, node `timestep`,
`root_time / count`.

Do not change the environment exports: they are the original sweep environment,
and substituting the production one costs 13-79 % per timestep.
