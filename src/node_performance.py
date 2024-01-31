import random
import math
from src.config import NodeType

class NodePerformance:
    def __init__(self, num_cpu_cores, num_gpu_compute_units, node_type, seed=0):
        self.cpu_power_model = None
        self.gpu_power_model = None
        self.cpu_performance_model = None
        self.gpu_performance_model = None
        self.node_type = node_type
                
        self.cpu_core_logical = round(num_cpu_cores)
        self.cpu_core_physical = round(num_cpu_cores/2)
        self.gpu_core = round(num_gpu_compute_units)
        
        random.seed(seed)
                  
        self.idle_cpu_consumption = random.randint(20, 70)
        self.idle_cpu_performance = random.randint(20, 70)
        self.max_cpu_consumption = random.randint(4*self.cpu_core_physical+self.idle_cpu_consumption, 300) # we assume a max CPU TDP of 300W
        self.max_cpu_performance = random.randint(7*self.cpu_core_physical+self.idle_cpu_performance, 14*self.cpu_core_physical+self.idle_cpu_performance) # we assume a max CPU performance of 1200 GFLOPS (see https://www.cpubenchmark.net/cpu_list.php) 
        
        self.idle_gpu_consumption = random.randint(20, 70)
        self.idle_gpu_performance = random.randint(20, 70)
        self.max_gpu_consumption = random.randint(3*self.gpu_core+self.idle_gpu_consumption, 200)
        self.max_gpu_performance = random.randint(round(1.5*self.gpu_core)+self.idle_gpu_performance, 1200)
        
        self.set_default_power_and_performance_models()

    def set_default_power_and_performance_models(self):
        # Set default power and performance models
        # https://www.desmos.com/calculator/yuwhv9aqjm?lang=it
        if self.node_type == NodeType.SERVER:
            self.a = random.uniform(2, 6)
            self.b = random.uniform(150, 180)
            self.c = random.uniform(0.5, 2.5)
        # https://www.desmos.com/calculator/xkzchgnurv?lang=it
        elif self.node_type == NodeType.DESKTOP:
            self.a = random.uniform(10, 20)
            self.b = random.uniform(12, 18)
            self.c = random.uniform(0.5, 2)
        # https://www.desmos.com/calculator/o2urtnfm54?lang=it
        elif self.node_type == NodeType.RASPBERRY:
            self.a = random.uniform(0.5, 1)
            self.b = random.uniform(2.5, 4.5)
        
        self.cpu_power_model = self.default_cpu_power_model
        self.gpu_power_model = self.default_gpu_power_model
        self.cpu_performance_model = self.default_cpu_performance_model
        self.gpu_performance_model = self.default_gpu_performance_model

    def compute_current_power_consumption_cpu(self, cpu_usage):
        return self.cpu_power_model(cpu_usage)
    
    def compute_current_performance_cpu(self, cpu_usage):
        return self.cpu_performance_model(cpu_usage)
    
    def compute_current_efficiency_cpu(self, cpu_usage):
        return self.compute_current_performance_cpu(cpu_usage) / self.compute_current_power_consumption_cpu(cpu_usage)  
    
    def compute_current_power_consumption_gpu(self, gpu_usage):
        return self.gpu_power_model(gpu_usage)
    
    def compute_current_performance_gpu(self, gpu_usage):
        return self.gpu_performance_model(gpu_usage)
    
    def compute_current_efficiency_gpu(self, gpu_usage):
        return self.compute_current_performance_gpu(gpu_usage) / self.compute_current_power_consumption_gpu(gpu_usage)
    
    def compute_current_power_consumption(self, cpu_usage, gpu_usage):
        cpu_power = self.compute_current_power_consumption_cpu(cpu_usage)
        gpu_power = self.compute_current_power_consumption_gpu(gpu_usage)
        
        return cpu_power + gpu_power
    
    def default_cpu_power_model(self, usage):
        if self.node_type == NodeType.RASPBERRY:
            return self.a * usage + self.b
        
        if usage <= self.cpu_core_physical:
            return self.a * usage + self.b
        else:
            return self.c * usage + (self.a * self.cpu_core_physical + self.b - self.c * self.cpu_core_physical)
    
    def server_cpu_performance_model(self, usage):
        if usage <= self.cpu_core_physical:
            return 7 * usage + self.idle_cpu_performance
        else:
            return ((self.max_cpu_performance - 7 * self.cpu_core_physical - self.idle_cpu_performance)/(self.cpu_core_physical)) * (usage - self.cpu_core_logical) + self.max_cpu_performance        
    
    def default_cpu_performance_model(self, usage):
        return 1
    
    def server_gpu_power_model(self, usage):
        return 0
        return 50 * math.log(usage + 5) +self.idle_gpu_consumption  # Example power model for GPU
    
    def default_gpu_power_model(self, usage):
        return 0
        return 50 * math.log(usage + 5) +self.idle_gpu_consumption

    def server_gpu_performance_model(self, usage):
        return 2 * math.log(usage + 1)  # Example performance model for GPU
    
    def default_gpu_performance_model(self, usage):
        return 1
        

if __name__ == "__main__":        
    num_cpu_cores = 56
    num_gpu_compute_units = 35
    # Example usage:
    my_device = NodePerformance(
        num_cpu_cores=num_cpu_cores,
        num_gpu_compute_units=num_gpu_compute_units,
    )

    # iterate from 1 to 28 cores and obtain the values for power and performance for cpu and save in an array
    cpu_power = []
    cpu_performance = []
    cpu_efficiency = []
    for i in range(1, num_cpu_cores+1):
        cpu_power.append(-my_device.compute_current_power_consumption_cpu(i))
        cpu_performance.append(my_device.compute_current_performance_cpu(i))
        cpu_efficiency.append(my_device.compute_current_efficiency_cpu(i))
        
    # do the same for gpu
    gpu_power = []
    gpu_performance = []
    gpu_efficiency = []
    for i in range(1, num_gpu_compute_units+1):
        gpu_power.append(my_device.compute_current_power_consumption_gpu(i))
        gpu_performance.append(my_device.compute_current_performance_gpu(i))
        gpu_efficiency.append(my_device.compute_current_efficiency_gpu(i))
        
    #plot using matplotlib in four subplots
    import matplotlib.pyplot as plt
    fig, axs = plt.subplots(2, 3)
    axs[0, 0].plot(cpu_power)
    axs[0, 0].set(xlabel='CPU Usage', ylabel='Power (W)')
    axs[0, 1].plot(cpu_performance, 'tab:orange')
    axs[0, 1].set(xlabel='CPU Usage', ylabel='Performance (GFLOPS)')
    axs[1, 0].plot(gpu_power, 'tab:green')
    axs[1, 0].set(xlabel='GPU Usage', ylabel='Power (W)')
    axs[1, 1].plot(gpu_performance, 'tab:red')
    axs[1, 1].set(xlabel='GPU Usage', ylabel='Performance (GFLOPS)')

    # add plot for cpu and gpu efficiency
    axs[0, 2].plot(cpu_efficiency, 'tab:blue')
    axs[0, 2].set(xlabel='CPU Usage', ylabel='Efficiency (GFLOPS/W)')
    axs[1, 2].plot(gpu_efficiency, 'tab:purple')
    axs[1, 2].set(xlabel='GPU Usage', ylabel='Efficiency (GFLOPS/W)')

    fig.tight_layout()
    plt.savefig('cpu_gpu_power_performance.png')

    # Results for our server w 14 physical cores and 28 logical cores
    # Power 57.52	60.88	63.3	65.17	66.52	67.6	68.71	69.77	70.66	72.79	74.7	77.47	80.05	82.99	86.44	90.32	94.26	98.41	102.87	107.25	111.89	116.37	120.35	124.25	128.02
    # Score 4022.4	4964.02	5889.33	6784.53	7656.63	8515.46	9350.99	10167.68	10963.89	11743.95	12477.77	12806.89	13125.6	13430.55	13730.05	14013.02	14302.21	14579.15	14833.48	15101.83	15367.51	15619.71	15822.55	16057.02	16269.12

