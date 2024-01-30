import math
from multiprocessing.managers import SyncManager
from multiprocessing import Process, Event, Manager, JoinableQueue
import time
import pandas as pd
import signal
import logging
import os
import sys

from src.network_topology import NetworkTopology
from src.topology import topo as LogicalTopology
from src.network_topology import  TopologyType
from src.utils import generate_gpu_types, NodeSupport
from src.node import node
from src.config import Utility, DebugLevel, SchedulingAlgorithm, ApplicationGraphType
import src.jobs_handler as job
import src.utils as utils
import src.plot as plot
from src.jobs_handler import message_data

class MyManager(SyncManager): pass

main_pid = ""
nodes_thread = []

def sigterm_handler(signum, frame):
    """Handles the SIGTERM signal by performing cleanup actions and gracefully terminating all processes."""
    # Perform cleanup actions here
    # ...    
    global main_pid
    if os.getpid() == main_pid:
        print("SIGINT received. Performing cleanup...")
        for t in nodes_thread:
            t.terminate()
            t.join()    
            
        print("All processes have been gracefully teminated.")
        sys.exit(0)  # Exit gracefully    

class Simulator_Plebiscito:
    def __init__(self, filename: str, n_nodes: int, node_bw: int, n_jobs: int, n_client: int, enable_logging: bool, use_net_topology: bool, progress_flag: bool, dataset: pd.DataFrame, alpha: float, utility: Utility, debug_level: DebugLevel, scheduling_algorithm: SchedulingAlgorithm, decrement_factor: float, split: bool, app_type: ApplicationGraphType) -> None:   
        self.filename = filename + "_" + utility.name + "_" + scheduling_algorithm.name + "_" + str(decrement_factor)
        if split:
            self.filename = self.filename + "_split"
        else:
            self.filename = self.filename + "_nosplit"
            
        self.n_nodes = n_nodes
        self.node_bw = node_bw
        self.n_jobs = n_jobs
        self.n_client = n_client
        self.enable_logging = enable_logging
        self.use_net_topology = use_net_topology
        self.progress_flag = progress_flag
        self.dataset = dataset
        self.debug_level = debug_level
        self.counter = 0
        self.alpha = alpha
        self.scheduling_algorithm = scheduling_algorithm
        self.decrement_factor = decrement_factor
        self.split = split
        self.app_type = app_type
        self.utility = utility
        
        self.job_count = {}
        
        self.nodes = []
        self.gpu_types = generate_gpu_types(n_nodes)

        # Set up the environment
        self.setup_environment()
        
    def startup_nodes(self):
         # create a suitable network topology for multiprocessing 
        MyManager.register('NetworkTopology', NetworkTopology)
        self.manager = MyManager()
        self.manager.start()
        
         #Build Topolgy
        self.t = LogicalTopology(func_name='ring_graph', max_bandwidth=self.node_bw, min_bandwidth=self.node_bw/2,num_clients=self.n_client, num_edges=self.n_nodes)
        self.network_t = self.manager.NetworkTopology(self.n_nodes, self.node_bw, self.node_bw, group_number=4, seed=4, topology_type=TopologyType.FAT_TREE)
        
        for i in range(self.n_nodes):
            self.nodes.append(node(i, self.network_t, self.gpu_types[i], self.utility, self.alpha, self.enable_logging, self.t, self.n_nodes, self.progress_flag, use_net_topology=self.use_net_topology, decrement_factor=self.decrement_factor))
            
    def get_nodes(self):
        return self.nodes
            
    def setup_environment(self):
        """
        Set up the environment for the program.

        Registers the SIGTERM signal handler, sets the main process ID, and initializes logging.
        """
        
        signal.signal(signal.SIGINT, sigterm_handler)
        global main_pid
        main_pid = os.getpid()

        logging.addLevelName(DebugLevel.TRACE, "TRACE")
        logging.basicConfig(filename='debug.log', level=self.debug_level.value, format='%(message)s', filemode='w')

        logging.debug('Clients number: ' + str(self.n_client))
        logging.debug('Edges number: ' + str(self.n_nodes))
        logging.debug('Requests number: ' + str(self.n_jobs))
        
    def setup_nodes(self, terminate_processing_events, start_events, use_queue, manager, return_val, queues, progress_bid_events):
        """
        Sets up the nodes for processing. Generates threads for each node and starts them.
        
        Args:
        terminate_processing_events (list): A list of events to terminate processing for each node.
        start_events (list): A list of events to start processing for each node.
        use_queue (list): A list of events to indicate if a queue is being used by a node.
        manager (multiprocessing.Manager): A multiprocessing manager object.
        return_val (list): A list of return values for each node.
        queues (list): A list of queues for each node.
        progress_bid_events (list): A list of events to indicate progress of bid processing for each node.
        """
        global nodes_thread
        
        for i in range(self.n_nodes):
            q = JoinableQueue()
            e = Event() 
            
            queues.append(q)
            use_queue.append(e)
            
            e.set()

        #Generate threads for each node
        for i in range(self.n_nodes):
            e = Event() 
            e2 = Event()
            e3 = Event()
            return_dict = manager.dict()
            
            self.nodes[i].set_queues(queues, use_queue)
            
            p = Process(target=self.nodes[i].work, args=(e, e2, e3, return_dict))
            nodes_thread.append(p)
            return_val.append(return_dict)
            terminate_processing_events.append(e)
            start_events.append(e2)
            e3.clear()
            progress_bid_events.append(e3)
            
            p.start()
            
        for e in start_events:
            e.wait()
    
    def collect_node_results(self, return_val, jobs: pd.DataFrame, exec_time, time_instant, save_on_file):
        """
        Collects the results from the nodes and updates the corresponding data structures.
        
        Args:
        - return_val: list of dictionaries containing the results from each node
        - jobs: list of job objects
        - exec_time: float representing the execution time of the jobs
        - time_instant: int representing the current time instant
        
        Returns:
        - float representing the utility value calculated based on the updated data structures
        """
        # self.counter = 0
        # self.job_count = {}
        
        if time_instant != 0:
            for v in return_val: 
                nodeId = v["id"]
                for _, j in jobs.iterrows():
                    # if j["job_id"] not in self.nodes[nodeId].bids:
                    self.nodes[nodeId].bids[j["job_id"]] = v["bids"][j["job_id"]]
                    if j["job_id"] not in self.job_count:
                        self.job_count[j["job_id"]] = 0
                    self.job_count[j["job_id"]] = v["counter"][j["job_id"]]
                # for key in v["counter"]:
                #     if key not in self.job_count:
                #         self.job_count[key] = 0
                #     self.job_count[key] += v["counter"][key]
                #     self.counter += v["counter"][key]
                self.nodes[nodeId].updated_cpu = v["updated_cpu"]
                self.nodes[nodeId].updated_gpu = v["updated_gpu"]
                self.nodes[nodeId].updated_bw = v["updated_bw"]
                self.nodes[nodeId].gpu_type = v["gpu_type"]
        
        return utils.calculate_utility(self.nodes, self.n_nodes, self.counter, exec_time, self.n_jobs, jobs, self.alpha, time_instant, self.use_net_topology, self.filename, self.network_t, self.gpu_types, save_on_file)
    
    def terminate_node_processing(self, events):
        global nodes_thread
        
        for e in events:
            e.set()
            
        # Block until all tasks are done.
        for nt in nodes_thread:
            nt.join()
            
    def clear_screen(self):
        # Function to clear the terminal screen
        os.system('cls' if os.name == 'nt' else 'clear')

    def print_simulation_values(self, time_instant, processed_jobs, queued_jobs: pd.DataFrame, running_jobs, batch_size):
        print()
        print("Infrastructure info")
        print(f"Number of nodes: {self.n_nodes}")
        
        for t in set(self.gpu_types):
            count = 0
            for i in self.gpu_types:
                if i == t:
                    count += 1
            print(f"Number of {t.name} GPU nodes: {count}")
        
        print()
        print("Performing simulation at time " + str(time_instant) + ".")
        print(f"# Jobs assigned: \t\t{processed_jobs}/{len(self.dataset)}")
        print(f"# Jobs currently in queue: \t{len(queued_jobs)}")
        print(f"# Jobs currently running: \t{running_jobs}")
        print(f"# Current batch size: \t\t{batch_size}")
        print()
        NODES_PER_LINE = 4
        count = 0
        print("Node CPU resource usage")
        for n in self.nodes:
            if count == NODES_PER_LINE:
                count = 0
                print()
            print("Node{0} ({1}):\t{2:3.0f}%\t".format(n.id, n.gpu_type,(n.initial_cpu - n.updated_cpu)/n.initial_cpu*100), end=" |   ")
            count += 1
            #print(f"Node{n.id} ({n.gpu_type}):\t{(n.initial_gpu - n.updated_gpu)/n.initial_gpu*100}%   ", end=" | ")
        print()
        print()
        print("Jobs in queue stats for gpu type:")
        if len(queued_jobs) == 0:
            print("<no jobs in queue>")
        else:
            print(queued_jobs["gpu_type"].value_counts().to_dict())

            
    def print_simulation_progress(self, time_instant, job_processed, queued_jobs, running_jobs, batch_size):
        self.clear_screen()
        self.print_simulation_values(time_instant, job_processed, queued_jobs, running_jobs, batch_size) 
        
    def deallocate_jobs(self, progress_bid_events, queues, jobs_to_unallocate):
        if len(jobs_to_unallocate) > 0:
            for _, j in jobs_to_unallocate.iterrows():
                data = message_data(
                            j['job_id'],
                            j['user'],
                            j['num_gpu'],
                            j['num_cpu'],
                            j['duration'],
                            j['bw'],
                            j['gpu_type'],
                            deallocate=True,
                            split=self.split,
                            app_type=self.app_type
                        )
                for q in queues:
                    q.put(data)

            for e in progress_bid_events:
                e.wait()
                e.clear()       

    def skip_deconfliction(self, jobs): # :)
        if jobs.empty:
            return True
        
        dispatch = []

        dispatch_= True
        
        if self.split:
            node_gpu = 0
            node_cpu = 0
            
            for node in self.nodes:
                node_gpu += node.get_avail_gpu()  # Consider caching these values if they don't change
                node_cpu += node.get_avail_cpu()
        
        for _, row in jobs.iterrows():
            num_gpu = row['num_gpu']
            num_cpu = row['num_cpu']
            
            if self.split:
                if node_cpu >= num_cpu and num_gpu >= num_gpu:
                    return False
                    # dispatch.append(row)
            else:
                for node in self.nodes:           
                    if NodeSupport.can_host(NodeSupport.get_node_type(node.gpu_type), NodeSupport.get_node_type(row["gpu_type"])) and node.get_avail_cpu() >= num_cpu and node.get_avail_gpu() >= num_gpu:
                        # Append the row from jobs DataFrame
                        return False
                        # dispatch.append(row)
                        # break
        return True
        # return pd.DataFrame(dispatch) if dispatch else None

    def run(self):
        # Set up nodes and related variables
        global nodes_thread
        terminate_processing_events = []
        start_events = []
        progress_bid_events = []
        use_queue = []
        manager = Manager()
        return_val = []
        queues = []
        self.setup_nodes(terminate_processing_events, start_events, use_queue, manager, return_val, queues, progress_bid_events)

        # Initialize job-related variables
        self.job_ids=[]
        jobs = pd.DataFrame()
        running_jobs = pd.DataFrame()
        processed_jobs = pd.DataFrame()

        # Collect node results
        start_time = time.time()
        self.collect_node_results(return_val, pd.DataFrame(), time.time()-start_time, 0, save_on_file=True)
        
        time_instant = 1
        batch_size = 5
        jobs_to_unallocate = pd.DataFrame()
        unassigned_jobs = pd.DataFrame()
        assigned_jobs = pd.DataFrame()
        prev_job_list = pd.DataFrame()
        prev_running_jobs = pd.DataFrame()
        
        while True:
            start_time = time.time()
            
            # Extract completed jobs
            prev_running_jobs = running_jobs.copy(deep=True)
            jobs_to_unallocate, running_jobs = job.extract_completed_jobs(running_jobs, time_instant)
            
            # Deallocate completed jobs
            self.deallocate_jobs(progress_bid_events, queues, jobs_to_unallocate)
            self.collect_node_results(return_val, pd.DataFrame(), time.time()-start_time, time_instant, save_on_file=False)
            
            #self.collect_node_results(return_val, pd.DataFrame(), time.time()-start_time, time_instant, save_on_file=False)
            
            if time_instant%25 == 0:
                plot.plot_all(self.n_nodes, self.filename, self.job_count, "plot")
            
            # Select jobs for the current time instant
            new_jobs = job.select_jobs(self.dataset, time_instant)
            
            # Add new jobs to the job queue
            prev_job_list = jobs.copy(deep=True)
            jobs = pd.concat([jobs, new_jobs], sort=False)
            
            # Schedule jobs
            jobs = job.schedule_jobs(jobs, self.scheduling_algorithm)
            
            n_jobs = len(jobs)
            if prev_job_list.equals(jobs) and prev_running_jobs.equals(running_jobs):
                n_jobs = 0
            
            jobs_to_submit = job.create_job_batch(jobs, n_jobs)
            
            unassigned_jobs = pd.DataFrame()
            assigned_jobs = pd.DataFrame()
            
            # Dispatch jobs
            if len(jobs_to_submit) > 0: 
                start_id = 0
                while start_id < len(jobs_to_submit):
                    subset = jobs_to_submit.iloc[start_id:start_id+batch_size]

                    if self.skip_deconfliction(subset) == False:
                        job.dispatch_job(subset, queues, self.use_net_topology, self.split)

                        for e in progress_bid_events:
                            e.wait()
                            e.clear() 
                            
                        exec_time = time.time() - start_time
                    
                        # Collect node results
                        a_jobs, u_jobs = self.collect_node_results(return_val, subset, exec_time, time_instant, save_on_file=False)
                        assigned_jobs = pd.concat([assigned_jobs, a_jobs])
                        unassigned_jobs = pd.concat([unassigned_jobs, u_jobs])
                    
                        # Deallocate unassigned jobs
                        self.deallocate_jobs(progress_bid_events, queues, u_jobs)
                    else:
                        unassigned_jobs = pd.concat([unassigned_jobs, subset])
                        #print('ktm')
                        
                    start_id += batch_size
            
            # Assign start time to assigned jobs
            assigned_jobs = job.assign_job_start_time(assigned_jobs, time_instant)
            
            # Add unassigned jobs to the job queue
            jobs = pd.concat([jobs, unassigned_jobs], sort=False)  
            running_jobs = pd.concat([running_jobs, assigned_jobs], sort=False)
            processed_jobs = pd.concat([processed_jobs,assigned_jobs], sort=False)
                    
            self.collect_node_results(return_val, pd.DataFrame(), time.time()-start_time, time_instant, save_on_file=True)
            
            self.print_simulation_progress(time_instant, len(processed_jobs), jobs, len(running_jobs), batch_size)
            time_instant += 1

            # if len(assigned_jobs) == 0 and len(unassigned_jobs) != 0 and batch_size > 3:
            #     batch_size -= 1
            # elif len(assigned_jobs) > len(unassigned_jobs)/3 and batch_size < 8:
            #     batch_size += 1

            # Check if all jobs have been processed
            if len(processed_jobs) == len(self.dataset) and len(running_jobs) == 0 and len(jobs) == 0: # add to include also the final deallocation
                break
        
        # Collect final node results
        self.collect_node_results(return_val, pd.DataFrame(), time.time()-start_time, time_instant+1, save_on_file=True)
        
        self.print_simulation_progress(time_instant, len(processed_jobs), jobs, len(running_jobs), batch_size)
        
        # Terminate node processing
        self.terminate_node_processing(terminate_processing_events)

        # Save processed jobs to CSV
        processed_jobs.to_csv(self.filename + "_jobs_report.csv")

        # Plot results
        if self.use_net_topology:
            self.network_t.dump_to_file(self.filename, self.alpha)

        plot.plot_all(self.n_nodes, self.filename, self.job_count, "plot")

    