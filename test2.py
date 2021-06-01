
from confusion_test import confusion
import pandas as pd
import torch.autograd.profiler as profiler


import numpy as np
import cProfile

# pred_1 = "/data/jjia/ssc_scoring/LK_time2_17patients.csv"
#
# label_1 = "/data/jjia/ssc_scoring/ground_truth_17_patients.csv"

pred_1 = "/data/jjia/ssc_scoring/1405_16pats_pred.csv"
#
label_1 = "/data/jjia/ssc_scoring/observer_agreement/16_patients/ground_truth_16patients.csv"

# pred_1 = "/data/jjia/ssc_scoring/models_pos/193/valid_pred_world.csv"
# label_1 = "/data/jjia/ssc_scoring/models_pos/193/valid_world.csv"

df_label = pd.read_csv(label_1)
df_pred = pd.read_csv(pred_1)

for df in [df_label, df_pred]:
    if df.columns[0] == "ID":
        del df["ID"]
        del df["Level"]

if df_label.columns[0] not in ['L1_pos', 'L1', 'disext']:
    df_label = pd.read_csv(label_1, header=None)
    if len(df_label.columns) == 5:
        columns = ['L1', 'L2', 'L3', 'L4', 'L5']
    elif len(df_label.columns) == 3:
        columns = ['disext', 'gg', 'retp']
    else:
        columns = ['unknown']
    df_label.columns = columns

if df_pred.columns[0] not in ['L1_pos', 'L1', 'disext']:
    df_pred = pd.read_csv(pred_1, header=None)
    if len(df_pred.columns) == 5:
        columns = ['L1', 'L2', 'L3', 'L4', 'L5']
    elif len(df_pred.columns) == 3:
        columns = ['disext', 'gg', 'retp']
    else:
        columns = ['unknown']
    df_pred.columns = columns


label_np = df_label.to_numpy()
pred_np = df_pred.to_numpy()
diff = pred_np - label_np

mean = np.mean(diff)
std = np.std(diff)

confusion(label_1, pred_1, adap_markersize=1)

# with profiler.profile(use_cuda=True) as prof:
#     with profiler.record_function("model_inference"):
#         confusion(label_1, pred_1, adap_markersize=1)
#
# prof.export_chrome_trace("trace1.json")

print("finish")
