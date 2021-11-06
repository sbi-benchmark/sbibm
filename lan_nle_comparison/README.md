# Comparison between LANs and NLE on the simlpe drift-diffusion model

This is a short demonstration for how to reproduce the comparison between LANs and NLE.

We perform the comparison in (log) likelihood space, comparing the synthetic likelihoods
of LAN and NLE against the analytic likelihoods obtained from https://github.com/DrugowitschLab/DiffModels.jl. 

Additionally, we use MCMC via slice sampling to compare two approaches in posterior space. 
For this comparison we have added the DDM as a task in a framework developed for benchmarking 
simulation-based inference algorithms, `sbibm`. 

In general, the code relies on three repositories, [`sbi`](https://github.com/mackelab/sbi) for using NLE, 
[`sbibm`](https://github.com/sbi-benchmark/sbibm) for simulating the data and loading the LAN keras weights, 
and [`benchmarking-results`](https://github.com/sbi-benchmark/results/tree/main/benchmarking_sbi) for running the benchmark. 

## Comparison in likelihood space
For a demo of the likelihood comparison you find a jupyter notebook in this folder. For
executing the notebook locally perform the steps outlined below. 

```bash
# clone repo
git clone https://github.com/mackelab/sbibm.git
# switch to branch
cd sbibm
git checkout ddm-task
# install locally (e.g., in a new conda env)
pip install -e .
# install missing nflow dependency
pip install UMNN
# open the notebook at /lan_nle_comparison
```

## Comparison in posterior space using `sbibm`

For a general overview over the benchmarking suite see https://sbi-benchmark.github.io. 

To run the benchmark on your local machine, please follow the steps below.

- **optional**: create and activate a new conda environment
```bash
conda create -n ddmtest python=3.8
conda activate ddmtest
```

- clone and install `benchmarking-results` repo from https://github.com/sbi-benchmark/results/tree/main/benchmarking_sbi

```bash
git clone https://github.com/mackelab/results.git
cd results/benchmarking_sbi
git checkout ddm
pip install -r requirements.txt
cd ../..
```

- clone and install sbibm repo on the `ddm-task` branch

```bash
git clone https://github.com/mackelab/sbibm.git
cd sbibm
git checkout ddm-task
pip install -e .
cd ..
```

- run the benchmark

```bash
cd results/benchmarking_sbi
python run.py task=ddm task.num_observation=1 algorithm=lan
```

More details about how to run the benchmark can be found at https://github.com/sbi-benchmark/results/tree/main/benchmarking_sbi. 
