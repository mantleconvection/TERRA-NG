# LUMI-G strong-scaling sweep

Produces the LUMI-G line of Fig. 5 of the preprint.

37 standard-mode points, MT32 on 1 GCD to MT2048 on 4096 GCDs, 8 ranks per node
on AMD MI250X. Named `std_MT<level>_g<gcds>_std.sh`.

## Running

```
TERRANG_BIN=<path-to>/mantlecirculation \
  sbatch --account=<your-project> std_MT256_g64_std.sh
```

`TERRANG_CFG` defaults to `../config_scal_A3.toml` and `TERRANG_OUT` to a
directory named after the point. The Slurm log lands next to the script.

To read the results:

```
python3 collect_lumi.py <outroot> [published.csv]
```

It extracts the per-step time from each point's
`timer_trees/timer_tree_9.json` and, given a published CSV, prints the
comparison.

## Building on LUMI

Cray's GPU transport layer must be linked, or MPI aborts at startup with
"GPU_SUPPORT_ENABLED is requested, but GTL library is not linked". With the
LUMI/25.03, PrgEnv-amd, rocm/6.3.4 and craype-accel-amd-gfx90a modules:

```
cmake <source> -DCMAKE_BUILD_TYPE=RelWithDebInfo \
  -DCMAKE_CXX_COMPILER=/opt/rocm-6.3.4/bin/hipcc \
  -DKokkos_ENABLE_HIP=ON -DKokkos_ARCH_AMD_GFX90A=ON \
  -DKokkos_ENABLE_SERIAL=ON -DKokkos_ENABLE_ROCTHRUST=ON -DKokkos_ENABLE_HWLOC=OFF \
  -DMPI_CXX_LINK_FLAGS="-Wl,--whole-archive,-lhugetlbfs,--no-whole-archive" \
  -DCMAKE_EXE_LINKER_FLAGS="-L/opt/cray/pe/mpich/8.1.32/gtl/lib -lmpi_gtl_hsa -Wl,-rpath,/opt/cray/pe/mpich/8.1.32/gtl/lib"
make -j16 mantlecirculation
```

Check with `ldd mantlecirculation | grep gtl` before submitting.
