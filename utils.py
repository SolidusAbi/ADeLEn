from itertools import chain
import pandas as pd
import numpy as np

def generate_roc_df(roc_list:list) -> pd.DataFrame:
    ''' 
        Create a DataFrame from a list of roc curves
        Args:
        -----
            roc_list: list
                List of N roc curves where N is the number of iterations. It is a 
                list of tuples of the form (fpr, tpr), where fpr is the false positive
                rate and tpr is the true positive rate.
        Returns:
        --------
            roc_df: pd.DataFrame
                DataFrame with multiindex
    '''
    index_names = [
        list(map(lambda x: 'It {}'.format(x), np.repeat(np.arange(len(roc_list)), 2) + 1 )),
        ['FPR', 'TPR']*len(roc_list)
    ]
        
    tuples = list(zip(*index_names))
    index = pd.MultiIndex.from_tuples(tuples)
    roc_df = pd.DataFrame(chain.from_iterable(roc_list), index=index)
    return roc_df


def generate_multi_df(data:list, index_names:list) -> pd.DataFrame:
    index_names = [
        list(map(lambda x: 'It {}'.format(x), np.repeat(np.arange(len(data)), len(index_names)) + 1 )),
        index_names*len(data)
    ]

    tuples = list(zip(*index_names))
    index = pd.MultiIndex.from_tuples(tuples)
    return pd.DataFrame(chain.from_iterable(data), index=index)