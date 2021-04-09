# SSc score prediction

* `script.sh` is used to submit jobs to slurm
* `set_args.py` stores the super parameters. imported by `run.py`
* `run.py` is the main file to train/infer/continur_train networks for ssc score prediction.
* `records.csv` and `cp_records.csv` is the same, recording/tracking all experiments.
* `confusion.py` is used to get the confusin matrix, accuracy, weighted kappa, MAE, etc. to evaluate trained networks.
* `models` directory save the results of each experiments. ID of each experiment is from the `records.csv`.
* `slurmlogs` directory saves the output of training logs.
* `dataset` directory saves the dataset.
