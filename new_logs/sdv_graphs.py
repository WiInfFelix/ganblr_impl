import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from pathlib import Path

sns.set_theme(style='whitegrid')
sns.set_palette('Set2')

def create_sdv_graph(frame: pd.DataFrame, dataset_name: str):

    ganblr = frame[frame['Model'] == 'GANBLR']
    ctgan = frame[frame['Model'] == 'CTGAN']

    # create a figure with two subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

    sns.barplot(
        data=ganblr,
        x='Epochs',
        y='Value',
        hue='K',
        ax=ax1
    )

    sns.barplot(
        data=ctgan,
        x='Epochs',
        y='Value',
        ax=ax2
    )

    # set super title
    fig.suptitle(f'SDV scores for {dataset_name}')

    #plot titles
    ax1.set_title(f'GANBLR')
    ax2.set_title(f'CTGAN')

    ax1.set_ylabel('Score')
    ax2.set_ylabel('')

    # set axes to the same scale
    ax1.set_ylim(ax2.get_ylim())

    plt.tight_layout()

    plt.show()


SUPERSTORE = Path('log_20231116-072219.csv')
MUSHROOMS = Path('log_20231115-224934.csv')
CREDIT_RISK = Path('log_20231115-113140.csv')

sup_df = pd.read_csv(SUPERSTORE, delimiter=';')
mush_df = pd.read_csv(MUSHROOMS, delimiter=';')
credit_df = pd.read_csv(CREDIT_RISK, delimiter=';')

create_sdv_graph(sup_df, 'Superstore dataset')
create_sdv_graph(mush_df, 'Mushrooms dataset')
create_sdv_graph(credit_df, 'Credit risk dataset')
