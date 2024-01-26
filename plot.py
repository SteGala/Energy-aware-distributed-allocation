import pandas as pd
import matplotlib.pyplot as plt

filenames = ["prova_LGF_FIFO_0.2_split.csv", "prova2.csv"]
labels = {
    "prova_LGF_FIFO_0.2_split.csv": "Plebiscito",
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

plt.savefig("helllooooo")