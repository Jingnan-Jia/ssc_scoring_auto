# -*- coding: utf-8 -*-
# @Time    : 4/20/21 1:40 PM
# @Author  : Jingnan
# @Email   : jiajingnan2222@gmail.com
# -*- coding: utf-8 -*-
# @Time    : 3/19/21 11:22 PM
# @Author  : Jingnan
# @Email   : jiajingnan2222@gmail.com
import glob
import matplotlib
# Force matplotlib to not use any Xwindows backend.
matplotlib.use('Agg')

import pandas as pd
import numpy as np
import os
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from sklearn.metrics import cohen_kappa_score
import statsmodels.api as sm
import numpy as np
import matplotlib.pyplot as plt


label_file = "61_valid_label_jjia.csv"
pred_file = "61_valid_pred_jjia.csv"
df_label = pd.read_csv(label_file)
df_pred = pd.read_csv(pred_file)

label = df_label['L1'].to_numpy().reshape(-1, 1)
pred = df_pred['L1'].to_numpy().reshape(-1, 1)

f, ax = plt.subplots(1, figsize=(8, 5))
sm.graphics.mean_diff_plot(label, pred, ax=ax)
f.savefig('61_valid_all_levels_bland_altman.png')
plt.close(f)

