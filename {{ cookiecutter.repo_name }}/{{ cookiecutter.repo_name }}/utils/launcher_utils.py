""" launcher_utils.py.

    Helper functions to parse a config file

    Inspired by OncoNet:
        https://github.com/yala/OncoNet_Public

    TODO: Modify this launcher to define a set of jobs rather than the
    combinatorial itertation of job
"""
import hashlib

# Constnats
POSS_VAL_NOT_LIST = 'Flag {} has an invalid list of values: {}. Length of list must be >=1'


def md5(key: str) -> str:
    """md5.

    Args:
        key (str): string to be hasehd
    Returns:
        Hashed encoding of str
    """
    return hashlib.md5(key.encode()).hexdigest()

def parse_dispatcher_config(config: dict):
    """parse_dispatcher_config.

    Args:
        config (dict): Dict loaded from json config file. This dict will be
            of the structure:

            # yapf: disable
            {
                'search_space' : [{'flag' : [arg1_test, arg2_test], },],
                'available_gpus': []
            }
            # yapf: enable

            If search_space is not a list but a dict, it will be converted to a
            list
    returns:
        jobs: a list of flag strings each of which encaspulates one job.
            E.g.: --train --cuda --dropout=0.1 ...
        experiment_axes: Axes that the grid search is running over
    """

    jobs = [""]
    experiment_axes = []

    # List of possible search spaces
    search_spaces = config['search_space']

    # Support a list of search spaces, convert to length one list for backward compatiblity
    if not isinstance(search_spaces, list):
        search_spaces = [search_spaces]

    # All search spaces will be combined
    for search_space in search_spaces:
        # Go through the tree of possible jobs and enumerate into a list of jobs
        for ind, flag in enumerate(search_space):

            # Example of possible flag: "seed"
            possible_values = search_space[flag]

            # If we have more than 1 value to test, record this as an axes
            if len(possible_values) > 1:
                experiment_axes.append(flag)

            children = []
            # Ensure that each arg is a LIST
            if len(possible_values) == 0 or not isinstance(possible_values, list):
                raise Exception(POSS_VAL_NOT_LIST.format(flag, possible_values))

            # For each possible value of flag
            for value in possible_values:

                # parent_job contains all prevoiusly enumerated combinations of
                # this job;
                for parent_job in jobs:
                    # Add boolean flag
                    if isinstance(value, bool):
                        if value:
                            new_job_str = "{} --{}".format(parent_job, flag)
                        else:
                            new_job_str = parent_job
                    # Add list flag
                    elif isinstance(value, list):
                        val_list_str = " ".join([str(v) for v in value])
                        new_job_str = "{} --{} {}".format(
                            parent_job, flag, val_list_str)
                    # Add no flag
                    elif value is None:
                        new_job_str = "{}".format(parent_job)

                    # Add new flag
                    else:
                        new_job_str = "{} --{} {}".format(
                            parent_job, flag, value)
                    # Add this job to all the children
                    children.append(new_job_str)

            # Reset the parent jobs to process next axis of jobs
            jobs = children

    return jobs, experiment_axes
