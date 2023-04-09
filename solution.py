import pandas as pd
import numpy as np
import scipy.stats as stats

chat_id = 252926140 # Ваш chat ID, не меняйте название переменной

def solution(data1: np.ndarray, data2: np.ndarray = None) -> bool:
    if data2 is None:
        # Condition 1: two historical samples
        t, p = stats.ttest_ind(data1, data1, equal_var=True)
    else:
        # Conditions 2 and 3: one historical sample and one modified sample
        t, p = stats.ttest_ind(data1, data2, equal_var=True)
    
    alpha = 0.07
    reject_null = p < alpha
    
    return reject_null
