import pandas as pd
from sklearn.model_selection import train_test_split
def statmodels_split(df, stratify=None, **kwargs):
    """
    Inputs
    df: pandas dataframe.
        if stratify is None, target column MUST be the first column in the dataframe
        
    stratify: target column or None
    
    Returns: 
    Tuple of dataframes (df_train, df_test) 
    """

    if stratify is None:
        y, X = df.iloc[:,0], df.drop(columns=df.columns[0])
        X_train, X_test, y_train, y_test = train_test_split(X,y, **kwargs)
    else:
        y, X = stratify, df.drop(columns = stratify.name)
        X_train, X_test, y_train, y_test = train_test_split(X,y,stratify=y, **kwargs)
    
    return pd.concat([X_train, y_train], axis=1), pd.concat([X_test, y_test], axis=1)