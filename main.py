import copy
from src.simulator import Simulator_Plebiscito
from src.config import Utility, DebugLevel, SchedulingAlgorithm, ApplicationGraphType
from src.dataset_builder import generate_dataset
from tst.brute_force_scheduler import BruteForceScheduler
from tst.kubernetes_scheduler import KubernetesScheduler
from plot import plot_consumption
import sys

if __name__ == '__main__':
    n_jobs = 1000
    dataset = generate_dataset(entries_num=n_jobs)
    
    simulator_0 = Simulator_Plebiscito(filename="0",
                        n_nodes=15,
                        node_bw=1000000000,
                        n_jobs=n_jobs,
                        n_client=3,
                        enable_logging=False,
                        use_net_topology=False,
                        progress_flag=False,
                        dataset=dataset,
                        alpha=1,
                        utility=Utility.POWER,
                        debug_level=DebugLevel.INFO,
                        scheduling_algorithm=SchedulingAlgorithm.FIFO,
                        decrement_factor=0,
                        split=True,
                        app_type=ApplicationGraphType.LINEAR,)
    
    simulator_2 = copy.deepcopy(simulator_0)
    simulator_2.filename = "2_" + "_".join(simulator_0.filename.split("_")[1:])
    
    simulator_4 = copy.deepcopy(simulator_0)
    simulator_4.filename = "4_" + "_".join(simulator_0.filename.split("_")[1:])
    
    simulator_6 = copy.deepcopy(simulator_0)
    simulator_6.filename = "6_" + "_".join(simulator_0.filename.split("_")[1:])
    
    simulator_8 = copy.deepcopy(simulator_0)
    simulator_8.filename = "8_" + "_".join(simulator_0.filename.split("_")[1:])

    simulator_0.set_outlier_number(0)
    simulator_0.startup_nodes()
    simulator_0.run()
    
    simulator_2.set_outlier_number(2)
    simulator_2.startup_nodes()
    simulator_2.run()
    
    simulator_4.set_outlier_number(4)
    simulator_4.startup_nodes()
    simulator_4.run()
    
    simulator_6.set_outlier_number(6)
    simulator_6.startup_nodes()
    simulator_6.run()
    
    simulator_8.set_outlier_number(8)
    simulator_8.startup_nodes()
    simulator_8.run()
    
    nodes = simulator_0.get_nodes()
    plot_consumption(nodes)
    
    simulator_kubernetes = KubernetesScheduler(nodes, dataset, "kubernetes", ApplicationGraphType.LINEAR, True)
    simulator_kubernetes.run()
    
    simulator_brute_force = BruteForceScheduler(nodes, dataset, "brute-force", ApplicationGraphType.LINEAR, True)
    simulator_brute_force.run()
    