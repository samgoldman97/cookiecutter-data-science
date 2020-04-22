""" launch_experiments.py.

    Code to train other models from config files

    Inspired by Onconet with original source:
    https://github.com/yala/OncoNet_Public/blob/master/scripts/dispatcher.py

    Usage:
    python scripts/launch_experiments.py --python-file-name training/train_model.py --experiment-config-file static/train_config.py --log-dir results/test/
    python scripts/launch_experiments.py --python-file-name training/train_model.py --experiment-config-file static/train_config.py --log-dir results/test/ --sbatch-script scripts/slurm_scripts/slurm_train_fastas.sh
"""
import sys
import os
import argparse
import json
import multiprocessing
import subprocess
from typing import Tuple
from enzpred.utils import launcher_utils

# Constants
RESULTS_PATH_APPEAR_ERR = "save_dir should not appear in config. It will be determined automatically per job"


def get_args():
    """ Get arguments """
    options = argparse.ArgumentParser()
    # Name of python file to run
    options.add_argument('--python-file-name',
                         action="store",
                         help="Name of python script to run!")
    options.add_argument('--experiment-config-file',
                         action="store",
                         help="Json config file")
    options.add_argument('--log-dir',
                         action="store",
                         help="Where to store results")
    options.add_argument("--sbatch-script",
                         action="store",
                         default=None,
                         help="Path to the sbatch script to launch programs")
    options.add_argument("--use-gpu",
                         action="store",
                         default=None,
                         help="Use this flag if launching on gpu")

    return options.parse_args()


## Multiprocessing


def launch_experiment(log_dir: str, python_file_name: str, gpu: int,
                      flag_string: str) -> Tuple[str, str]:
    """launch_experiments.

    Launch an experiment and direct logs and results to a unique filepath.
    Alert of something goes wrong.

    Args:
        log_dir (str): Directory for the log of the results
        python_file_name (str): Name of the python file passed
        gpu (int): gpu to run this machine on.
        flag_string (str): flags to use for this model run. Will be fed into
            scripts/main.py

    Returns:
        Tuple[str,str]: results_path, log_path

    """
    if not os.path.isdir(log_dir):
        os.makedirs(log_dir)

    log_name = launcher_utils.md5(flag_string)
    log_stem = log_name

    results_path = os.path.join(log_dir, "{}".format(log_stem))
    log_path = os.path.join(log_dir, "{}.txt".format(log_stem))

    if gpu:
        experiment_string = "CUDA_VISIBLE_DEVICES={} python -u {} {} --out {}".format(
            gpu, python_file_name, flag_string,
            os.path.join(results_path, "out"))
    else:
        experiment_string = "python -u {} {} --out {}".format(
            python_file_name, flag_string, os.path.join(results_path, "out"))

    # Redirect both stdout and output to a file for now
    shell_cmd = "{} > {} 2>&1".format(experiment_string, log_path)

    if not os.path.exists(results_path):
        print("Launched exp: {}".format(shell_cmd))
        os.makedirs(results_path)
        subprocess.call(shell_cmd, shell=True)
    else:
        print("ERROR LAUNCHING; dir {} exists".format(results_path))

    return results_path, log_path


def worker(log_dir: str, python_file_name: str, gpu: int,
           job_queue: multiprocessing.Queue, done_queue: multiprocessing.Queue):
    """worker.

    Worker thread for each gpu. Consumes all jobs and pushes results to done_queue.

    Args:
        log_dir (str): Directory for the log of the results
        python_file_name (str): Name of the python file passed
        gpu (int): Gpu this worker can access
        job_queue (multiprocessing.Queue): Queue of available jobs
        done_queue (multiprocessing.Queue): Queue where to push resulst
    """
    while not job_queue.empty():
        params = job_queue.get()
        if params is None:
            return
        done_queue.put(launch_experiment(log_dir, python_file_name, gpu, params))


def multiprocessing_launch(log_dir: str, python_file_name: str,
                           experiment_config_file: str, use_gpu: bool,
                           **kwargs):
    """multiprocessing_launch.

    Use this to launch jobs on multiprocessing

    Args:
        log_dir (str): log_dir
        python_file_name (str): python_file_name
        experiment_config_file (str): experiment_config_file
        use_gpu (bool): use_gpu
        kwargs: Absorb slurm script arg
    """

    experiment_config = json.load(open(experiment_config_file, 'r'))

    if 'save_dir' in experiment_config['search_space']:
        print(RESULTS_PATH_APPEAR_ERR)
        sys.exit(1)

    job_list, experiment_axes = launcher_utils.parse_dispatcher_config(
        experiment_config)

    # For multiprocessing:
    job_queue = multiprocessing.Queue()
    done_queue = multiprocessing.Queue()

    # Add jobs to the queue
    for job in job_list:
        job_queue.put(job)

    if use_gpu:
        print("Launching Dispatcher with {} jobs!".format(len(job_list)))
        for gpu in experiment_config['available_gpus']:
            print("Start gpu worker {}".format(gpu))
            multiprocessing.Process(target=worker,
                                    args=(log_dir, python_file_name, gpu, job_queue,
                                          done_queue)).start()

    else:
        print("Launching Dispatcher with {} jobs!".format(len(job_list)))
        multiprocessing.Process(target=worker,
                                args=(log_dir, python_file_name, None, job_queue,
                                      done_queue)).start()


## Slurm
def slurm_launch(log_dir: str, python_file_name: str, experiment_config_file: str,
                 use_gpu: bool, sbatch_script: str, **kwargs):
    """slurm_launch.

    Use this to launch on slurm

    Args:
        log_dir (str): log_dir
        python_file_name (str): python_file_name
        experiment_config_file (str): experiment_config_file
        use_gpu (bool): use_gpu
        sbatch_script (str): sbatch_script
        kwargs: kwargs
    """

    experiment_config = json.load(open(experiment_config_file, 'r'))
    if 'save_dir' in experiment_config['search_space']:
        print(RESULTS_PATH_APPEAR_ERR)
        sys.exit(1)

    job_list, experiment_axies = launcher_utils.parse_dispatcher_config(
        experiment_config)

    for flag_string in job_list:
        if not os.path.isdir(log_dir):
            os.makedirs(log_dir)

        log_stem = launcher_utils.md5(flag_string)
        results_path = os.path.join(log_dir, "{}".format(log_stem))

        # Useful for slurm?
        log_path = os.path.join(log_dir, "{}.txt".format(log_stem))
        shell_cmd = "python -u {} {} --out {}".format(
            python_file_name, flag_string, os.path.join(results_path, "out"))

        # Run in sbatch
        if sbatch_script is not None:
            shell_cmd = ("sbatch --export=CMD=\"{}\" {}".format(
                shell_cmd, sbatch_script))

        if not os.path.exists(results_path):
            os.makedirs(results_path)

            subprocess.call(shell_cmd, shell=True)
        else:
            raise Exception(
                "Path to this results file {} already exists".format(
                    results_path))

        print("Launched exp: {}\n".format(shell_cmd))


if __name__ == "__main__":
    args = vars(get_args())

    if args['sbatch_script'] is not None:
        slurm_launch(**args)
    else:
        multiprocessing_launch(**args)
