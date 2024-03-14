import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

datasets_v2 = ['bn_5_v2', 'bn_10_v2', 'bn_20']
datasets_v1 = ['cassini', 'pawelczyk', 'two_sines']
samples = 500

for dataset in datasets_v1:
    df = pd.read_csv('synthetic/{}.csv'.format(dataset))
    df = df.sample(samples, replace=False)
    sns.scatterplot(data=df, x='x1', y='x2', hue='y')
    plt.savefig('visualizations/{}.pdf'.format(dataset))
    plt.savefig('visualizations/{}.png'.format(dataset))
    plt.close()
    print('plotted {}'.format(dataset))

samples = 250
for dataset in datasets_v2:
    df = pd.read_csv('synthetic_v2/{}.csv'.format(dataset))
    df = df.sample(samples, replace=False)
    sns.pairplot(df, hue='y')
    plt.savefig('visualizations/{}.pdf'.format(dataset))
    plt.savefig('visualizations/{}.png'.format(dataset))
    plt.close()
    print('plotted {}'.format(dataset))