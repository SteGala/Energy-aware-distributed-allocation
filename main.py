import copy
from src.simulator import Simulator_Plebiscito
from src.config import Utility, DebugLevel, SchedulingAlgorithm, ApplicationGraphType
from src.dataset_builder import generate_dataset
from tst.brute_force_scheduler import BruteForceScheduler
from plot import plot_consumption
import sys

if __name__ == '__main__':
    n_jobs = 30
    dataset = generate_dataset(entries_num=n_jobs)
    
    simulator_lgf = Simulator_Plebiscito(filename="1",
                        n_nodes=10,
                        node_bw=1000000000,
                        n_jobs=n_jobs,
                        n_client=3,
                        enable_logging=False,
                        use_net_topology=False,
                        progress_flag=False,
                        dataset=dataset,
                        alpha=1,
                        utility=Utility.LGF,
                        debug_level=DebugLevel.INFO,
                        scheduling_algorithm=SchedulingAlgorithm.FIFO,
                        decrement_factor=0,
                        split=True,
                        app_type=ApplicationGraphType.LINEAR,)
    
    simulator_power = copy.deepcopy(simulator_lgf)
    simulator_power.filename = "2"
    simulator_power.utility = Utility.POWER

    simulator_lgf.startup_nodes()
    simulator_lgf.run()
    simulator_power.startup_nodes()
    simulator_power.run()
    
    nodes = simulator_lgf.get_nodes()
    
    simulator_brute_force = BruteForceScheduler(nodes, dataset, "prova2", ApplicationGraphType.LINEAR, True)
    
    simulator_brute_force.run()
    