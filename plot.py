import pandas as pd
import matplotlib.pyplot as plt

if __name__ == "__main__":
    filenames = ["brute-force.csv", "0_POWER_FIFO_0_split.csv", "2_POWER_FIFO_0_split.csv", "4_POWER_FIFO_0_split.csv", "kubernetes.csv"]
    labels = {
        "brute-force.csv": "Brute Force",
        "0_POWER_FIFO_0_split.csv": "Plebiscito-POWER-0",
        "2_POWER_FIFO_0_split.csv": "Plebiscito-POWER-2",
        "4_POWER_FIFO_0_split.csv": "Plebiscito-POWER-4",
        "kubernetes.csv": "Kubernetes"
    }
    data = {}
    
    fig, (ax1, ax2) = plt.subplots(1, 2)
    

    for filename in filenames:
        df = pd.read_csv(filename)
        df = df.filter(regex=(".*consumption"))
        
        ax1.plot([i for i in range(len(df))], list(df.sum(axis=1)), label=labels[filename])
        data[labels[filename]] = list(df.sum(axis=1))
    
    max_len = max([len(v) for v in data.values()])  
    plot_data = []
    
    for i in range(max_len):
        d = []
        for l in labels:
            if len(data[labels[l]]) > i:
                d.append(data[labels[l]][i])
            else:
                d.append(0)
        plot_data.append(d)
            
    df = pd.DataFrame(plot_data, columns=list(labels.values()))
    _ = df.boxplot(ax=ax2, rot=90)
    
    ax1.legend()
    ax1.set_xlabel("Time")
    ax1.set_ylabel("Power Consumption (W)")
    #fig.xlabel("Time")
    #fig.ylabel("Power Consumption")
    plt.tight_layout()
    fig.savefig("helllooooo")

def plot_consumption(nodes):
    distinct_node_type = []
    for n in nodes:
        if n.gpu_type not in distinct_node_type:
            distinct_node_type.append(n.gpu_type)
    
    fig, axs = plt.subplots(len(distinct_node_type))
        
    for n in nodes:
        cpu_power = []
        cpu_performance = []
        cpu_efficiency = []
        gpu_power = []
        gpu_performance = []
        gpu_efficiency = []
    
        for i in range(1, n.initial_cpu+1):
            cpu_power.append(n.performance.compute_current_power_consumption_cpu(i))
            cpu_performance.append(n.performance.compute_current_performance_cpu(i))
            cpu_efficiency.append(n.performance.compute_current_efficiency_cpu(i))
        
        x = [i for i in range(1, n.initial_cpu+1)]
        id = 0
        for id2, t in enumerate(distinct_node_type):
            if n.gpu_type == t:
                id = id2
                break
            
        axs[id].plot(x, cpu_power, label=f"{n.id}")
            
        # do the same for gpu
        for i in range(1, n.initial_gpu+1):
            gpu_power.append(n.performance.compute_current_power_consumption_gpu(i))
            gpu_performance.append(n.performance.compute_current_performance_gpu(i))
            gpu_efficiency.append(n.performance.compute_current_efficiency_gpu(i))

    fig.tight_layout()
    fig.savefig('node_power_consumption.png')