import random
import sys
import time
import numpy as np
import pandas as pd
from src.config import SchedulingAlgorithm, ApplicationGraphType

def assign_job_start_time(dataset: pd.DataFrame, time_instant):
    dataset.replace(-1, time_instant, inplace=True)
    return dataset
        
def extract_completed_jobs(dataset: pd.DataFrame, time_instant):
    if len(dataset) == 0:
        return dataset, dataset
    
    condition = dataset.exec_time + dataset.duration < time_instant
    ret = dataset[condition]
    
    if len(ret) > 0:
        dataset = dataset[~condition]
    
    return ret, dataset

def select_jobs(dataset, time_instant):
    return dataset[dataset['submit_time'] == time_instant]

def create_job_batch(dataset, batch_size):
    ret = dataset.head(batch_size)
    dataset.drop(index=dataset.index[:batch_size], axis=0, inplace=True)
    return ret

def schedule_jobs(jobs: pd.DataFrame, scheduling_algorithm: SchedulingAlgorithm):
    if scheduling_algorithm == SchedulingAlgorithm.FIFO:
        return jobs.sort_values(by=["submit_time"])
    elif scheduling_algorithm == SchedulingAlgorithm.SDF:
        return jobs.sort_values(by=["duration"])

def dispatch_job(dataset: pd.DataFrame, queues, use_net_topology=False, split=True, app_type=ApplicationGraphType.LINEAR):        
    if use_net_topology:
        timeout = 1 # don't change it
    else:
        timeout = 0.05

    for _, job in dataset.iterrows():
        data = message_data(
                    job['job_id'],
                    job['user'],
                    job['num_gpu'],
                    job['num_cpu'],
                    job['duration'],
                    job['bw'],
                    job['gpu_type'],
                    deallocate=False,
                    split=split,
                    app_type=app_type
                )
        
        for q in queues:
            q.put(data)

        time.sleep(timeout)

def get_simulation_end_time_instant(dataset):
    return dataset['submit_time'].max() + dataset['duration'].max()

def generate_application_graph(layer_number, app_type, bandwidth):
    graph = np.zeros((layer_number, layer_number))
    
    for i in range(layer_number):
        for j in range(i):
            if app_type == ApplicationGraphType.LINEAR:
                if j == i-1:
                    #b = random.uniform(0.5, 1.5)*bandwidth
                    b = bandwidth
                    graph[i][j] = b
                    graph[j][i] = b
            else:
                prob = 0
                if app_type == ApplicationGraphType.GRAPH20:
                    prob = 0.2
                elif app_type == ApplicationGraphType.GRAPH40:
                    prob = 0.4
                elif app_type == ApplicationGraphType.GRAPH60:
                    prob = 0.6
                
                #b = np.random.choice([0, 1], p=[1-prob, prob])*random.uniform(0.5, 1.5)*bandwidth
                b = np.random.choice([0, 1], p=[1-prob, prob])*bandwidth
                graph[i][j] = b
                graph[j][i] = b
                
    return graph        

def message_data(job_id, user, num_gpu, num_cpu, duration, bandwidth, gpu_type, deallocate=False, split=True, app_type=ApplicationGraphType.LINEAR):
    
    random.seed(job_id)
    np.random.seed(int(job_id))
    
    if split:
        layer_number = random.choice([1, 2, 3, 4, 5])
    else:
        layer_number = 1

    # use numpy to create an array of random numbers with length equal to the number of layers. As a constraint, the sum of the array must be equal to the number of GPUs
    NN_gpu = np.random.dirichlet(np.ones(layer_number), size=1)[0] * num_gpu
    NN_cpu = np.random.dirichlet(np.ones(layer_number), size=1)[0] * num_cpu
    NN_data_size = generate_application_graph(layer_number, app_type, 1000000)

    if split:
        max_layer_bid = layer_number
        min_layer_bid = 1
    else:
        max_layer_bid = layer_number
        min_layer_bid = layer_number

    bundle_size = 2
    
    data = {
        "job_id": int(),
        "user": int(),
        "num_gpu": int(),
        "num_cpu": int(),
        "duration": int(),
        "N_layer": len(NN_gpu),
        "N_layer_min": min_layer_bid, # Do not change!! This could be either 1 or = to N_layer_max
        "N_layer_max": max_layer_bid,
        "N_layer_bundle": bundle_size, 
        "edge_id":int(),
        "NN_gpu": NN_gpu,
        "NN_cpu": NN_cpu,
        "NN_data_size": NN_data_size,
        "gpu_type": gpu_type,
        }

    data['edge_id']=None
    data['job_id']=job_id
    data['user']=user
    data['num_gpu']=num_gpu
    data['num_cpu']=num_cpu
    data['duration']=duration
    data['job_id']=job_id
    
    if deallocate:
        data["unallocate"] = True

    return data
