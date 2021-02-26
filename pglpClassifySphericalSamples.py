import tensorflow as tf
import pandas as pd
import numpy as np
import os
import sys
import re


#debugging
from pdb import set_trace as pause
pd.options.display.max_colwidth = 150
pd.options.display.max_rows = 3000

ct_shape = (256, 512, 4)

samples_dir = sys.argv[1]
output_file = sys.argv[2]


sample_files = [os.path.join(samples_dir, file) for file in os.listdir(samples_dir) if file.endswith('.npy')]


samples_df = pd.DataFrame()
samples_df['file'] = sample_files

samples_df['id'] = [os.path.basename(file).split("-aug")[0] for file in samples_df['file']]

samples_df['lung'] = [1]*len(samples_df)
samples_l2 = samples_df.copy()
samples_l2['lung'] = [2]*len(samples_l2)

samples_df = samples_df.append(samples_l2.copy(), ignore_index=True)
samples_df = samples_df.sort_values('file')
samples_df.reset_index(inplace=True, drop=True)


print(samples_df)

x = np.zeros((len(samples_df), ct_shape[0], ct_shape[1], 2))

for index, row in samples_df.iterrows():
    if row['lung'] == 1:
        x[index] = np.load(row['file'])[:,:,[0,2]]
    if row['lung'] == 2:
        x[index] = np.load(row['file'])[:,:,[1,3]]


model = tf.keras.models.load_model('ct_classifier.h5')

y_pred = model.predict(x)
y_pred = y_pred[:, 0]

samples_df['covid_prob'] = y_pred



per_subject_results = pd.DataFrame()
per_subject_results['id'] = list(set(samples_df['id']))
per_subject_results = per_subject_results.set_index('id', drop=True)



covid_probs = []
for subject in per_subject_results.index:
    covid_probs.append(samples_df.loc[samples_df['id'] == subject]['covid_prob'].mean())
per_subject_results['covid_prob'] = covid_probs



class_dict = {0: 'others', 1: 'covid'}

per_subject_results['pred_class'] = [class_dict[pred] for pred in np.round(per_subject_results['covid_prob'])]

per_subject_results.to_csv(output_file)














