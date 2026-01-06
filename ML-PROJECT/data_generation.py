import numpy as np
import pandas as pd
from simulation import run_full_model

# Number of samples
N_samples = 2000

data = []

for _ in range(N_samples):
    D = np.random.uniform(1, 5)
    H = np.random.uniform(6, 20)
    G = np.random.uniform(0.1, 1.0)
    L = np.random.uniform(0.2, 1.2)
    C_NaOH0 = np.random.uniform(700, 1800)
    k_caus = np.random.uniform(0.1, 1.0)

    CO2_tpy, cost, eff = run_full_model(D, H, G, L, 0.12, C_NaOH0,
                                       20, 3, k_caus, 0.85,
                                       0.1, 150)
    data.append([D,H,G,L,C_NaOH0,k_caus,cost,eff])

df = pd.DataFrame(data, columns=["D","H","G","L","C_NaOH0","k_caus","cost","efficiency"])
df.to_csv("training_data.csv", index=False)
print("✅ Training data saved to training_data.csv")
