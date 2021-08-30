# CHEKING DEPENDENCY BETWEEN Electrical and Others

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


def Check_Depend(X: pd.DataFrame, HUE: str, feat: str):
    if X[feat].dtype.kind in 'biufc':
        facet = sns.FacetGrid(X, hue=HUE, aspect=4, margin_titles=True)
        facet.map(sns.kdeplot, feat, shade=True)
        facet.set(xlim=(0, X[feat].max()))
        facet.add_legend()
    else:
        Classes = X[feat].unique().tolist()
        Values = [X[X[feat] == c][HUE].value_counts() for c in Classes]
        df = pd.DataFrame(Values)
        df.index = Classes
        df.plot(kind='bar', stacked=True, figsize=(10, 5))

    plt.show()
