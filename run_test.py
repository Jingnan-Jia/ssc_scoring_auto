# -*- coding: utf-8 -*-
# @Time    : 3/6/21 10:08 AM
# @Author  : Jingnan
# @Email   : jiajingnan2222@gmail.com
import argparse
import csv

import psutil
import os
import os
def has_handle(fpath):
    process = psutil.process_iter()
    process_ls = list(process)
    for proc in process_ls:
        try:
            fff = proc.open_files()
            for item in fff:
                if fpath == item.path:
                    return True
        except Exception:
            pass

    return False

# current_abs_dir = os.path.dirname(os.path.realpath(__file__))  # abosolute path of the current .py file
# file_fpath = current_abs_dir+"/t.csv"
# a = has_handle(file_fpath)
#
# if not a:
#     with open(file_fpath, 'a') as csv_file:
#         writer = csv.writer(csv_file, delimiter=',')
#         writer.writerow('ID')

from filelock import FileLock


lockfile = r"t.csv"
lock = FileLock(lockfile + ".lock")
log_dict = {}
with lock:
    import pandas as pd

    record_file = 'record4.csv'
    lock = FileLock(record_file + ".lock")
    new_id = None
    with lock:  # with this lock,  open a file for exclusive access
        with open(record_file, 'a') as csv_file:
            if not os.path.isfile(record_file) or os.stat(record_file).st_size == 0:  # empty?
                new_id = 1
                df = pd.DataFrame()
            else:
                df = pd.read_csv(record_file)
                last_id = df.tail(1)['ID'].values[0]
                new_id = last_id + 1
            df = df.astype(object)  # df need to be object, otherwise NAN cannot live in it

    for index, row in df.iterrows():
        if 'CPU_utilized' not in list(df.columns) or row['CPU_utilized'] in [None, np.nan]:
            jobid = row['outfile'].split('-')[-1].split('_')[0]  # extract job id from outfile name
            seff = os.popen('seff ' + jobid)  # get job information
            for line in seff.readlines():
                line = line.split(': ')  # must have space to be differentiated from time format 00:12:34
                if len(line) == 2:
                    key, value = line
                    key = '_'.join(key.split(' '))  # change 'CPU utilized' to 'CPU_utilized'
                    value = value.split('\n')[0]
                else:
                    pass  # throw unusable results
                log_dict.update({key: value})

    print(' yes')

