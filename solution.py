import pandas as pd
import numpy as np
import scipy.stats as stats

chat_id = 252926140 # Ваш chat ID, не меняйте название переменной

def solution(sample_x: np.ndarray, sample_y: np.ndarray = None, condition: int = 1) -> bool:
    if condition == 1:
        # Sample distribution X and Y are historical data
        # Sample size is 500
        # Limitation on the frequency of wrong decision: 0.101
        alpha = 0.07
        n_x = n_y = 500
    elif condition == 2:
        # Sample distribution X is historical data
        # Sample distribution Y is modified type 2 of H0
        # Sample size is 500
        # Limitation on the frequency of wrong decision: 0.0196
        alpha = 0.0196
        n_x = n_y = 500
    elif condition == 3:
        # Sample distribution X is historical data
        # Sample distribution Y is modified type 1 of H1
        # Sample size is 500
        # Restriction on the frequency of wrong decision: 0.00280
        alpha = 0.0028
        n_x = n_y = 500

    # Calculate the sample means and standard deviations
    x_bar = sample_x.mean()
    y_bar = sample_y.mean() if sample_y is not None else x_bar
    s_x = sample_x.std(ddof=1)
    s_y = sample_y.std(ddof=1) if sample_y is not None else s_x

    # Calculate the pooled standard deviation and test statistic
    s_p = np.sqrt(((n_x - 1) * s_x ** 2 + (n_y - 1) * s_y ** 2) / (n_x + n_y - 2))
    t = (x_bar - y_bar) / (s_p * np.sqrt(1/n_x + 1/n_y))

    # Calculate the p-value
    p_value = 2 * (1 - stats.t.cdf(abs(t), df=n_x + n_y - 2))

    # Reject the null hypothesis if p-value is less than
