import pandas as pd
import matplotlib.pyplot as plt

if __name__ == "__main__":
    filenames = ["1_LGF_FIFO_0_split.csv", "2.csv", "prova2.csv"]
    labels = {
        "1_LGF_FIFO_0_split.csv": "Plebiscito-CPU",
        "2.csv": "Plebiscito-POWER",
        "prova2.csv": "Brute Force"
    }
    data = {}

    for filename in filenames:
        df = pd.read_csv(filename)
        df = df.filter(regex=(".*consumption"))
        
        plt.plot([i for i in range(len(df))], list(df.sum(axis=1)), label=labels[filename])
        
    # data_df = pd.DataFrame(data)
    # data_df.plot.line()
    plt.legend()
    plt.xlabel("Time")
    plt.ylabel("Power Consumption")

    plt.savefig("helllooooo")

def plot_consumption(nodes):
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
        plt.plot(x, cpu_power, label=f"{n.id}")
            
        # do the same for gpu
        for i in range(1, n.initial_gpu+1):
            gpu_power.append(n.performance.compute_current_power_consumption_gpu(i))
            gpu_performance.append(n.performance.compute_current_performance_gpu(i))
            gpu_efficiency.append(n.performance.compute_current_efficiency_gpu(i))

    plt.tight_layout()
    plt.savefig('cpu_gpu_power_performance.png')