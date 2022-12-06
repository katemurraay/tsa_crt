import numpy as np
def r_squared(true, predicted):
    y = np.array(true)
    y_hat = np.array(predicted)
    y_m = y.mean()
    ss_res = np.sum((y-y_hat)**2)
    ss_tot = np.sum((y-y_m)**2)
    
    return 1 - (ss_res/ss_tot)