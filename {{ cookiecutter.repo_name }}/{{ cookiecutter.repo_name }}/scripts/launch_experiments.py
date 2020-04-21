'''
    launch_experiments.py
    
    Code to train other models from config files
    Taken from: 
    https://github.com/yala/OncoNet_Public/blob/master/scripts/dispatcher.py
    Usage: 
    python scripts/launch_experiments.py --python-file-name training/train_model.py --experiment-config-file static/train_config.py --log-dir results/test/
    python scripts/launch_experiments.py --python-file-name training/train_model.py --experiment-config-file static/train_config.py --log-dir results/test/ --sbatch-script scripts/slurm_scripts/slurm_train_fastas.sh 
'''
import os
from os.path import dirname, realpath
import sys
sys.path.append(dirname(dirname(realpath(__file__))))

from utils.launcher_utils import parse_dispatcher_config, md5
import argparse
import json 
import multiprocessing
import subprocess

# Constants 
RESULTS_PATH_APPEAR_ERR = 'save_dir should not appear in config. It will be determined automatically per job'

def get_args(): 
    options = argparse.ArgumentParser()
    # Name of python file to run
    options.add_argument('--python-file-name', action="store",
                         help="Name of python script to run!")
    options.add_argument('--experiment-config-file', action="store",
                         help="Json config file")
    options.add_argument('--log-dir', action="store",
                         help="Where to store results")
    options.add_argument("--sbatch-script", action="store", default=None,
                         help="Path to the sbatch script to launch programs") 
                         
    return options.parse_args()

## Multiprocessing

def launch_experiment(args, gpu, flag_string):
    '''
    Launch an experiment and direct logs and results to a unique filepath.
    Alert of something goes wrong.
    :gpu: gpu to run this machine on.
    :flag_string: flags to use for this model run. Will be fed into
    scripts/main.py
    '''
    if not os.path.isdir(args.log_dir):
        os.makedirs(args.log_dir)

    log_name = md5(flag_string)
    log_stem = log_name

    results_path = os.path.join(args.log_dir,"{}".format(log_stem))
    log_path = os.path.join(args.log_dir,"{}.txt".format(log_stem))

    experiment_string = "CUDA_VISIBLE_DEVICES={} python -u {} {} --save-dir {}".format(
        gpu, args.python_file_name, flag_string, results_path)

    # Redirect both stdout and output to a file for now
    shell_cmd = "{} > {} 2>&1".format(experiment_string, log_path)
    print("Launched exp: {}".format(shell_cmd))

    if not os.path.exists(results_path):
        os.makedirs(results_path)
        subprocess.call(shell_cmd, shell=True)
    else: 
        print("ERROR LAUNCHING; dir {} exists".format(results_path))


    return results_path, log_path

def worker(args, gpu, job_queue, done_queue):
    '''
    Worker thread for each gpu. Consumes all jobs and pushes results to done_queue.
    :gpu - gpu this worker can access.
    :job_queue - queue of available jobs.
    :done_queue - queue where to push results.
    '''
    while not job_queue.empty():
        params = job_queue.get()
        if params is None:
            return
        done_queue.put(
            launch_experiment(args, gpu, params))


def multiprocessing_launch(args):
    ''' Use this to launch jobs on multiprocessing '''

    experiment_config = json.load(open(args.experiment_config_file, 'r'))

    if 'save_dir' in experiment_config['search_space']:
        print(RESULTS_PATH_APPEAR_ERR)
        sys.exit(1)

    job_list, experiment_axies = parse_dispatcher_config(experiment_config)

    # For multiprocessing: 
    job_queue = multiprocessing.Queue()
    done_queue = multiprocessing.Queue()
    
    # Add jobs to the queue  
    for job in job_list:
        job_queue.put(job)
    print("Launching Dispatcher with {} jobs!".format(len(job_list)))
    for gpu in experiment_config['available_gpus']:
        print("Start gpu worker {}".format(gpu))
        multiprocessing.Process(target=worker, args=(args, gpu, job_queue, done_queue)).start()

## Slurm
def slurm_launch(args):
    ''' Use this to launch on slurm'''

    experiment_config = json.load(open(args.experiment_config_file, 'r'))
    if 'save_dir' in experiment_config['search_space']:
        print(RESULTS_PATH_APPEAR_ERR)
        sys.exit(1)

    job_list, experiment_axies = parse_dispatcher_config(experiment_config)

    for flag_string in job_list:
        if not os.path.isdir(args.log_dir):
            os.makedirs(args.log_dir)


        log_stem = md5(flag_string)
        results_path = os.path.join(args.log_dir,"{}".format(log_stem))

        # Useful for slurm?
        log_path = os.path.join(args.log_dir,"{}.txt".format(log_stem))
        shell_cmd = "python -u {} {} --save-dir {}".format(
            args.python_file_name, flag_string, results_path)

        # Run in sbatch
        if args.sbatch_script is not None: 
            shell_cmd = ("sbatch --export=CMD=\"{}\" {}"
                              .format(shell_cmd, args.sbatch_script)
                              )
            
        if not os.path.exists(results_path):
            os.makedirs(results_path)
            
            subprocess.call(shell_cmd, shell=True)
        else:
            raise Exception("Path to this results file {} already exists"
                            .format(results_path)
                            )

        print("Launched exp: {}\n".format(shell_cmd))

if __name__=="__main__":
    args = get_args()

    # multiprocessing_launch(args)
    slurm_launch(args)

