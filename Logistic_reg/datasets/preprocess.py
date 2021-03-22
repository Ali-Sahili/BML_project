import numpy as np
import pandas as pd

def preprocess_data(filename)
    file_ = open("colon-cancer","r")
    data = [a.strip() for a in file_.readlines()]
    print("data size: ",len(data))

    res = np.asarray([[a.split(':')[-1] for a in b.split()] for b in data])
    file_.close()

    pd.DataFrame(res).to_csv(filename + '.csv')
