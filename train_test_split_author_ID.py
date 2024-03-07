#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import sys
import pathlib
import random
import pandas as pd
from sklearn.model_selection import GroupShuffleSplit

##get list of sessions containing text type
def check_author_type(a_df,auth_type='user'):
  set_of_sessions = set(a_df['session_id'])
  print('total sessions for author type %s: %d'%(auth_type,len(set_of_sessions)))
  session_list = []
  for sesh in set_of_sessions:
    sub_by_session = a_df.loc[a_df['session_id'] == sesh].copy()
    auth_samples = sub_by_session.loc[sub_by_session['sentence_author'] == auth_type].copy()
    if len(auth_samples) > 0:
      session_list.append(sesh)
  return session_list

def filtering_preprocess(a_df,min_sessions,filt_sesh=True,filt_auth=True):
  a_df = a_df.loc[a_df['sentence_text'] != 'NO_DATA'].copy()
  if filt_sesh == True:
    has_user_data = set(check_author_type(a_df, auth_type='user'))
    has_api_data = set(check_author_type(a_df, auth_type='api'))
    has_edit_data = set(check_author_type(a_df, auth_type='user_and_api'))
    keep_data = has_user_data.intersection(has_edit_data,has_api_data)
    print('sessions satisfy all conditions %s'%(len(keep_data)))
    a_df = a_df[a_df.session_id.isin(keep_data)]
  #filter to make sure there is enough data per author
  if filt_auth == True:
    author_list = set(a_df['author_id'])
    remove_list = []
    for auth in author_list:
      sub_df = a_df.loc[a_df['author_id'] == auth].copy()
      num_sesh = len(set(sub_df['session_id']))
      if num_sesh < min_sessions:
        remove_list.append(auth)
    a_df = a_df[~a_df.author_id.isin(remove_list)]
  return a_df

def prep_detect_author(a_df, SEED=42):
  random.seed(SEED)
  set_of_total_labels = sorted(list(set(a_df['author_id'])))
  list_of_data = []
  for author in set_of_total_labels:
    curr_auth = a_df.loc[a_df['author_id'] == author].copy()
    list_of_data.append(curr_auth)
  curr_usable_df = pd.concat(list_of_data, ignore_index=True)
  return curr_usable_df

def generate_sequence_dataframes(train_df, test_df, num_labels=3, num_trials=20, seed=6):
  random.seed(seed)
  df_list = []
  new_thresh=5
  seq_of_trials = random.sample(list(range(0,666)),k=num_trials)
  labels_for_choice = []
  set_of_labels = set(test_df['label'])
  for lbl in set_of_labels:
    sub_df = test_df.loc[test_df['label'] == lbl].copy()
    if len(sub_df.index) >= new_thresh:
      labels_for_choice.append(lbl)
  print('%d auth with at least %d examples in test' % (len(labels_for_choice), new_thresh))
  for idx in range(len(seq_of_trials)):
    baby_seed = seq_of_trials[idx]
    random.seed(baby_seed)
    subset_of_labels = random.sample(labels_for_choice,num_labels)
    sub_train_df = train_df[train_df.label.isin(subset_of_labels)]
    list_for_test = []
    for lbl in subset_of_labels:
      sub_df = test_df.loc[test_df['label'] == lbl].copy()
      samp_sub_df = sub_df.sample(n=len(sub_df), random_state=baby_seed)
      list_for_test.append(samp_sub_df)
    sub_test_df = pd.concat(list_for_test,ignore_index=True)
    df_list.append((sub_train_df, sub_test_df))
  return df_list

#####################################################################
#Edit parameters here
#####################################################################
coauthor_data = pd.read_csv('CoAuthor_Data_segment.csv')
data_dir='EDIT for saved data directory'

#reproducibility seed
RANDOM = 422

#minimum number of sessions required for all authors
min_sess = 10

#number of trials for train and test
num_trials=20

########################################################################
#filtering total data to satisfy the minimum number of sessions condition
coauthor_data_filtered = filtering_preprocess(coauthor_data, min_sess)
print("%d authors with at least %d sessions" % (len(set(coauthor_data_filtered['author_id'])), min_sess))

the_df = prep_detect_author(coauthor_data_filtered)

#create ineger IDs for each author ID
id2label = {}
label2id = {}
cnt = 0
for auth in sorted(list(set(the_df['author_id']))):
  id2label[cnt] = auth
  label2id[auth] = cnt
  cnt += 1
labels = [label2id[auth] for auth in the_df['author_id']]
the_df['label'] = labels
gs = GroupShuffleSplit(n_splits=1, test_size=.2, random_state=42)
train_idx, test_idx = next(gs.split(the_df, the_df['label'], groups=the_df['prompt_id']))

#intermediate save of filtered data
X_save_train = the_df.loc[train_idx].copy()
X_save_test = the_df.loc[test_idx].copy()
X_save_train.to_csv('%s/coauthor_%s_train_for_auth_ID.csv'%(data_dir,granularity),index=False)
X_save_test.to_csv('%s/coauthor_%s_test_for_auth_ID.csv'%(data_dir,granularity),index=False)

#save data for each text type combination and number of authors/labels
text_type_list = ['user_text', 'user_api_text', 'user_edit_text', 'user_api_edit_text']
save_data_out = '%s/auth_detect/%d_labels_train_'
for NUM_LBLS in [3, 4, 5, 7, 11]:
  num_lbls = min(NUM_LBLS, len(set(X_save_train['label'])))
  for text_type in text_type_list:
    list_of_usable_df = generate_sequence_dataframes(X_save_train, X_save_test, num_labels=num_lbls, num_trials=num_trials, seed=RANDOM)
    parent_save_dir = '%s/auth_detect' % (data_dir)
    save_data_out = '%s/%d_labels_train_%s' % (parent_save_dir,num_lbls,text_type)
    #create directory and any parents if necessary
    pathlib.Path(save_data_out).mkdir(parents=True, exist_ok=True)
    for idx in range(len(list_of_usable_df)):
      df_pair = list_of_usable_df[idx]
      train_df = df_pair[0]
      test_df = df_pair[1]
      train_df.to_csv('%s/train_%d.csv'%(save_data_out,idx))
      test_df.to_csv('%s/test_%d.csv'%(save_data_out,idx))


