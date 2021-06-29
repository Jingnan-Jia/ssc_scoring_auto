import numpy as np
import pandas as pd
import os
import csv

class Path:
    def __init__(self, id, check_id_dir=False) -> None:
        self.id = id  # type: int
        self.slurmlog_dir = 'slurmlogs'
        self.model_dir = 'models'
        self.data_dir = 'dataset'

        self.id_dir = os.path.join(self.model_dir, str(int(id)))  # +'_fold_' + str(args.fold)
        if check_id_dir:  # when infer, do not check
            if os.path.isdir(self.id_dir):  # the dir for this id already exist
                raise Exception('The same id_dir already exists', self.id_dir)

        for dir in [self.slurmlog_dir, self.model_dir, self.data_dir, self.id_dir]:
            if not os.path.isdir(dir):
                os.makedirs(dir)
                print('successfully create directory:', dir)

        self.model_fpath = os.path.join(self.id_dir, 'model.pt')
        self.model_wt_structure_fpath = os.path.join(self.id_dir, 'model_wt_structure.pt')

    def label(self, mode: str):
        return os.path.join(self.id_dir, mode + '_label.csv')

    def pred(self, mode: str):
        return os.path.join(self.id_dir, mode + '_pred.csv')

    def pred_int(self, mode: str):
        return os.path.join(self.id_dir, mode + '_pred_int.csv')

    def pred_end5(self, mode: str):
        return os.path.join(self.id_dir, mode + '_pred_int_end5.csv')

    def loss(self, mode: str):
        return os.path.join(self.id_dir, mode + '_loss.csv')

    def data(self, mode: str):
        return os.path.join(self.id_dir, mode + '_data.csv')


id_16_pats = [26, 28, 29, 49, 66, 70, 77, 140, 170, 179, 203, 209, 210, 227, 238, 263]

id_16_pats = ["026", "028", "029", "049", '066', "070", "077", "140", "170", "179", "203", "209", "210", "227", "238", "263"]

# ex_1044_fold1: valid:179, 227, 263, test: 28, 49, 66, 70, 210, 238
# ex_1043_fold2: valid:140,
# ex_1045_fold3: valid:203, 209,
# ex_1046_fold4: valid:26, 29, 77, 170
ex_id_dict = {1:1405,
              2:1404,
              3:1411,
              4:1410}

pat_id_dict = {1:['179', '227', '263'],
              2: ['140'],
              3: ['026', '029', '077', '170'],
              4: ['203', '209']}

def data_fpath(ex_id_dict):
    test_path = Path(ex_id_dict[1]).pred_end5('test')
    path_dict = {test_path: ['028', '049', '066', '070', '210', '238']}
    for fold, ex_id in ex_id_dict.items():
        valid_path = Path(ex_id).pred_end5('valid')
        path = {valid_path:pat_id_dict[fold]}
        path_dict.update(path)

    return path_dict

pred_fpath_dt: dict = data_fpath(ex_id_dict)  # len = 5, test, 1,2,3,4 fold
id_ls_, pred_ls_ = [], []
for level in [1,2,3,4,5]:
    pred_dt = {}
    for pred_path, pat_list in pred_fpath_dt.items():
        if 'test' in os.path.basename(pred_path):
            data_name_path = os.path.dirname(pred_path) + '/test_data.csv'
        else:
            data_name_path = os.path.dirname(pred_path) + '/valid_data.csv'
        df = pd.read_csv(data_name_path, header=None)
        # df.insert(0, 'row_num', range(0, len(df)))  # here you insert the row count
        df_first_c = df.loc[:, 0]
        index_ls = []
        for pat_id in pat_list:
            for index, row in df_first_c.iteritems():
                pat_name = 'Pat_' + pat_id
                if pat_name in row:
                        if 'Level' + str(level) + "_middle" in row:
                            index_ls.append(index)
                            break
        df_pred = pd.read_csv(pred_path)
        pred_ls = []
        for idx in index_ls:
            pred = df_pred.iloc[idx].to_numpy()  # np.array shape: [3,]
            pred_ls.append(pred)
        for pat_id, pred in zip(pat_list, pred_ls):
            pred_dt.update({pat_id: pred})


    print(pred_dt)
    id_ls = []
    pred_ls = []
    for id, pred in pred_dt.items():
        pred_ls.append(pred)
        id_ls.append(int(id))

    id_ls, pred_ls = zip(*sorted(zip(id_ls, pred_ls)))
    id_ls_.extend(id_ls)
    pred_ls_.extend(pred_ls)

if not os.path.isfile(str(ex_id_dict[1])+'_16pats_pred.csv'):
    with open(str(ex_id_dict[1])+'_16pats_pred.csv', 'w') as csv_file:
        writer = csv.writer(csv_file, delimiter=',')
        head = ['disext', 'gg', 'retp']
        writer.writerow(head)

with open(str(ex_id_dict[1])+'_16pats_pred.csv', 'a') as f:
    csvwriter = csv.writer(f)
    for pred, id in zip(pred_ls_, id_ls_):
        row = [id]
        row.extend(pred)
        csvwriter.writerow(pred)

with open(str(ex_id_dict[1])+'_16pats_id.csv', 'w') as f:
    csvwriter = csv.writer(f)
    for pred in id_ls_:
        csvwriter.writerow([pred])




