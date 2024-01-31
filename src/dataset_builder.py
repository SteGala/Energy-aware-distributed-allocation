import csv
import pandas as pd
import numpy as np
import os
import random
import math

path = os.getcwd()
dataset = path + '/df_dataset.csv'

def generate_dataset(entries_num = 100):
    """
    Generate a new dataset with the specified number of entries.
    
    Args:
    - entries_num (int): The number of entries to generate.
    
    Returns:
    - pandas.DataFrame: A new dataset with the specified number of entries.
    """
    jobs = init_go(num_jobs=entries_num)
    df = pd.DataFrame(jobs)
    return df

# function from Alibaba's trace
def set_job_list_arrival_time(job_list, arrival_rate=None, interval=60, shuffle_order=False):
    """
    job_list: jobs to execute in this run
    arrival_rate: num of jobs to arrive at each time interval (-1 or None means no changes)
    interval: time interval (default: 60)
    shuffle_order: bool, whether each user's inherent job order are shuffled (default: False)
    """
    if arrival_rate is None or arrival_rate < 0:
        return 0  # respect the original submit time
    if shuffle_order is True:
        np.random.shuffle(job_list)
    else:
        job_list.sort(key=lambda e: (e.get('submit_time', float('inf')), e['job_id']))

    arrival_counter = 0
    for job in job_list:
        arrival_time = (arrival_counter // arrival_rate) + 1 # * interval
        job['submit_time'] = arrival_time
        arrival_counter += 1
    
    return job_list

# function from Alibaba's trace
def _add_job(job_list, job_dict, describe_dict=None):
    # Add job (job_dict) into job_list
    for key, value in job_dict.items():
        if value is not None and value.isdigit() and key != 'user':
            if type(value) == str:
                job_dict[key] = round(float(value))
            else:  # duration becomes an int
                job_dict[key] = round(value)
        elif key in ['wait_time','user_dur','user_gpu_dur','group_dur','group_gpu_dur']:
            try:
                job_dict[key] = float(value)
            except:
                pass

    keys = ['num_cpu', 'num_gpu', 'submit_time', 'num_inst']
    for key in keys:
        if key not in job_dict or job_dict[key] == '':
            if key in ['num_cpu', 'num_gpu']:
                job_dict[key] = 0
            else:  # key in ['submit_time', 'num_inst']
                job_dict[key] = 1
        else:
            if key in ['num_cpu', 'num_gpu']:  # in %
                job_dict[key] = float(job_dict[key]) / 3 #round(100 * float(job_dict[key]))
            else:
                job_dict[key] = round(float(job_dict[key]))
        if key == 'num_inst':
            job_dict[key] = 1


    # Add entries to be used in scheduling
    job_dict['duration'] = random.randint(1, 50) # job_dict['duration'] = int(job_dict['duration'])
    if job_dict['duration'] == 0:
        job_dict['duration'] = 1  # fix duration == 0 problem.
    job_dict['size'] = int((job_dict['num_gpu'] + job_dict['num_cpu']) * job_dict['duration']) # (gpu + cpu) x duration
    job_dict['on_time'] = 0
    job_dict['wasted'] = 0
    job_dict['jct'] = -1
    job_dict['resource'] = [job_dict['num_gpu'], job_dict['num_cpu']] # list of resources
    job_dict['node'] = None
    job_dict["exec_time"] = -1
    job_dict["bw"] = float(job_dict["write_count"])
    job_dict["final_node_allocation"] = []
    job_dict["final_gpu_allocation"] = []
    job_dict["deadline"] = job_dict['submit_time'] + job_dict['duration'] * (1 + 0.1 * random.random()) # 10% deadline slack

    # Add duration estimation
    if describe_dict is not None:
        jd_user = describe_dict.get(job_dict['user'])
        if jd_user is not None:
            job_dict['dur_avg'] = float(jd_user['mean'])  # expectation
            job_dict['dur_std'] = float(jd_user['std'])  # standard deviation
            job_dict['dur_med'] = float(jd_user['50%'])  # median
            job_dict['dur_trim_mean'] = float(jd_user['trim_mean'])  # discard 10% top and 10% tail when calc. mean

    # Remove original unused entries
    for drop_col in ['fuxi_job_name','fuxi_task_name','inst_id','running_cluster','model_name','iterations','interval','vc','jobid','status']:
        if drop_col in job_dict: job_dict.pop(drop_col)
        
    job_dict['gpu_type'] = 'SERVER'
    
    if job_dict['num_cpu'] < 20:
        job_list.append(job_dict)
    
# function from Alibaba's trace
def add_job(csv_file, describe_dict, limit=None):
    """
    limit: To avoid reading too many jobs when the sampled number << total number of jobs in trace file.
    """
    job_list = []
    with open(csv_file, 'r') as fd:
        reader = csv.DictReader(fd, delimiter=',')
        keys = reader.fieldnames
        for i, row in enumerate(reader):
            _add_job(job_list, row, describe_dict)
            # if limit is not None and i >= 1000:
            #     break
    return job_list

# function from Alibaba's trace
def init_go(num_jobs=100):
    cur_time = 0
    arrivals = 1
    random.seed(0)
    job_list = add_job(dataset, None, limit=num_jobs)
    if (num_jobs is not None) and num_jobs <= len(job_list):
        #random.shuffle(job_list)

        job_list = job_list[:num_jobs]
    job_list = set_job_list_arrival_time(job_list, arrivals)

    return job_list



