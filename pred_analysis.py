import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv(f'./submission/Epoch_22_prediction_id_17_reproduce_base_17_ensemble.csv')

plt.figure(figsize = (20, 10))
plt.subplot(1, 2, 1)
plt.scatter(range(len(df['fake'])), df['fake'], alpha = 0.025)
plt.title(f'Preds | Label : [Fake]')
plt.subplot(1, 2, 2)
plt.scatter(range(len(df['real'])), df['real'], alpha = 0.025)
plt.title(f'Preds | Label : [Real]')
plt.show()