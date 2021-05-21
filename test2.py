
from confusion import confusion

# pred_1 = "/data/jjia/ssc_scoring/LK_time2_17patients.csv"
#
# label_1 = "/data/jjia/ssc_scoring/ground_truth_17_patients.csv"

pred_1 = "/data/jjia/ssc_scoring/1068_16pats_pred.csv"

label_1 = "/data/jjia/ssc_scoring/observer_agreement/16_patients/ground_truth_16patients.csv"

confusion(label_1, pred_1)
print("finish")
