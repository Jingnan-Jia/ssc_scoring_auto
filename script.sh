#!/bin/bash
#SBATCH --partition=gpu-long
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-gpu=6
#SBATCH -t 7-00:00:00
#SBATCH --mem-per-gpu=90G
#SBATCH --mail-type=end
#SBATCH --mail-user=jiajingnan2222@gmail.com

eval $(conda shell.bash hook)

conda activate py37

job_id=$SLURM_JOB_ID
echo $SLURM_JOB_ID
echo $job_id
echo $(hostname)

cp script.sh slurmlogs/slurm-${job_id}.sh

#idx=0; export CUDA_VISIBLE_DEVICES=$idx; stdbuf -oL python -u train.py 2>slurmlogs/slurm-${job_id}_$idx.err 1>slurmlogs/slurm-${job_id}_$idx.out --outfile=slurmlogs/slurm-${job_id}_$idx --hostname=$(hostname) --fold=1 --level=3 &
#idx=1; export CUDA_VISIBLE_DEVICES=$idx; stdbuf -oL python -u train.py 2>slurmlogs/slurm-${job_id}_$idx.err 1>slurmlogs/slurm-${job_id}_$idx.out --outfile=slurmlogs/slurm-${job_id}_$idx --hostname=$(hostname) --fold=1 --level=4 &
idx=0; export CUDA_VISIBLE_DEVICES=$idx; stdbuf -oL python -u run.py 2>slurmlogs/slurm-${job_id}_$idx.err 1>slurmlogs/slurm-${job_id}_$idx.out --outfile=slurmlogs/slurm-${job_id}_$idx --hostname=$(hostname) --fold=5 --level=0 --remark='resume fold5' &
#idx=1; export CUDA_VISIBLE_DEVICES=$idx; stdbuf -oL python -u run.py 2>slurmlogs/slurm-${job_id}_$idx.err 1>slurmlogs/slurm-${job_id}_$idx.out --outfile=slurmlogs/slurm-${job_id}_$idx --hostname=$(hostname) --fold=4 --level=0 --remark='resume fold4' &
#idx=2; export CUDA_VISIBLE_DEVICES=$idx; stdbuf -oL python -u run.py 2>slurmlogs/slurm-${job_id}_$idx.err 1>slurmlogs/slurm-${job_id}_$idx.out --outfile=slurmlogs/slurm-${job_id}_$idx --hostname=$(hostname) --fold=3 --level=0 --remark='resume fold3' &

wait





