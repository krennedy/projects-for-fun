#http://whiskyanalysis.com/index.php/database/

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

df = pd.read_csv('whiskeys.csv')

def drop_null_prices(df):
    """ Since we're trying to predict price, there is no point in
    considering entries which have no price.
    """
    idx = df.Cost.notnull()
    df = df[idx]
    return df

def transform_dollars_to_integers(df):
    """ The input type of the prices is dollars signs. Transform 
    these to integers.
    """
    df.loc[:, 'Cost'] = df['Cost'].apply(lambda s: len(s))
    return df

def clean_up_column_names(df):
    """ None of this multi-word nonsense!
    And let's keep it alphanumeric only
    """
    df = df.rename(columns = lambda s: s.replace(' ', '').lower())
    df = df.rename(columns = {'#': 'number'})
    return df

def extract_year(df):
    """ FIXME: actually do this
    If we can extract year 'yo' from the title, let's do it.
    Could also consider using the number of words in the title
    """
    pass

def do_some_plots(df):
    """ EDA, eill be deleted
    """
    plt.subplot(221)
    plt.scatter(df.number, df.metacritic, alpha=0.2)

    plt.subplot(222)
    sns.boxplot(data=df, x='cost', y='metacritic')

    plt.subplot(223)
    sns.violinplot(data=df, y='cost', x='country')

    

df = drop_null_prices(df)
df = transform_dollars_to_integers(df)
df = clean_up_column_names(df)

#do_some_plots(df)
# NEXT STEPS:
#   1. one-hot encoding
#   2. picking what method in scikit learn
#   3. Nice figures


plt.show()
