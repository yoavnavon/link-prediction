import numpy as np
import pandas as pd
from dynamobi import test_features

if __name__ == "__main__":
    size_train = 15000000
    size_test = 10000000
    feats_dim = 100
    print('Creating DataFrames')
    df_train = pd.DataFrame(np.random.randn(size_train,feats_dim))
    df_test = pd.DataFrame(np.random.randn(size_test,feats_dim))
    
    df_train['Class'] = np.random.randint(0,high=2,size=size_train)
    df_test['Class'] = np.random.randint(0,high=2,size=size_test)
    print('Testing')
    size, xgb_results, rf_results = test_features(df_train, df_test, print_results=True)
    