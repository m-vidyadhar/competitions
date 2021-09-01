import numpy as np
import pandas as pd

import matplotlib.pyplot as plt


def plt_classes(df, x_cols, l_col="labels") -> None:
    assert len(x_cols) == 2, "Only 2D plots are currently supported!"
    assert type(df) == pd.DataFrame, "Pass df as Pandas Dataframe!"

    labels = df.loc[:, l_col].unique()
    for label in labels:
        idx = df.loc[:, l_col] == label
        plt.scatter(df.loc[idx, x_cols[0]], df.loc[idx, x_cols[1]])
    
    plt.show()
    return


def plt_dec_boundary() -> None:
    return


def plt_classpreds() -> None:
    return