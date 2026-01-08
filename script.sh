#!/bin/bash
#SBATCH -N 1
#SBATCH -C gpu
#SBATCH -o logs/sfno%k.err
#SBATCH -e logs/sfno%j.err
#SBATCH -G 1
#SBATCH -q regular
#SBATCH -J sfno_test
#SBATCH --mail-user=yp1223@ic.ac.uk
#SBATCH --mail-type=ALL
#SBATCH -t 00:10:00
#SBATCH -A m4505

mkdir -p logs

source ~/.bashrc   
cd /global/homes/y/ypincha/blackholes/subgrid-blackhole
conda activate bh-sphere

python train.py --epochs 5 --batch_size 1 \
    --lr 0.001 --patience 3 --plot_freq 2 \
    --plot_dir "test_plots_${SLURM_JOB_ID}" \
    --device cuda