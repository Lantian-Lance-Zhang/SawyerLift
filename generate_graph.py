import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
import os

runs = []
logs_dir = 'evaluation_logs'

for run in os.listdir(logs_dir):
    print('Reading from', run)
    df = pd.read_csv(os.path.join(logs_dir, run))

    # This will probably not be used
    smoothing = 10  # moving average
    df['Smoothed'] = df['Value'].rolling(smoothing).sum() / smoothing
    print('Len:', len(df))

    runs.append(df)

rewards = np.vstack([run['Smoothed'] for run in runs])
mean = np.mean(rewards, axis=0)
std = np.std(rewards, axis=0)

sns.set(rc={'figure.figsize': (10, 5)})
sns.lineplot(y=mean, x=runs[0]['Step'])

plt.fill_between(
    x=runs[0]['Step'],
    y1=mean - std, y2=mean + std,
    alpha=0.2
)

plt.xlabel('Training Episodes')
plt.ylabel('Evaluation Return')
plt.show()
