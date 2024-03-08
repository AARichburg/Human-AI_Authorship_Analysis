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
    print('%s sessions satisfy all conditions'%(len(keep_data)))
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

def generate_session_data(a_df,SEED=42):
  list_of_sessions = list(set(a_df['session_id']))
  session_out_df = pd.DataFrame(columns=['example_id','session_id','author_id','prompt_id','all_text','user_text','user_api_text','user_edit_text','user_api_edit_text','user_label','user_api_label','user_edit_label','user_api_edit_label'])
  count = 0
  for sesh in list_of_sessions:
    sub_df = a_df.loc[a_df['session_id'] == sesh].copy()
    auth = list(sub_df['author_id'])[0]
    prompt = list(sub_df['prompt_id'])[0]
    user_text, user_api_text, user_edit_text, user_api_edit_text, all_text = reconstruct_session(sub_df)
    new_row = {'example_id': count, 'session_id':sesh, 'author_id':auth, 'prompt_id':prompt, 'all_text':all_text[0], 'user_text':user_text[0], 'user_api_text':user_api_text[0], 'user_edit_text':user_edit_text[0], 'user_api_edit_text':user_api_edit_text[0], 'user_text_label':user_text[1], 'user_api_label':user_api_text[1], 'user_edit_label':user_edit_text[1], 'user_api_edit_label':user_api_edit_text[1]}
    session_out_df = session_out_df._append(new_row, ignore_index=True)
    count+=1
  return session_out_df

#collects user/api/edited segments to generate session text in order
#can filter out any of the above categories
def reconstruct_session(a_df):
  user_df = a_df.loc[a_df['sentence_author'] == 'user'].copy()
  api_df = a_df.loc[a_df['sentence_author'] == 'api'].copy()
  edit_df = a_df.loc[a_df['sentence_author'] == 'user_and_api'].copy()
  all_text = combine_text(a_df)
  #smp_amt = min(math.ceil(stats.mean([len(user_df.index),len(api_df.index),len(edit_df.index)])),len(user_df.index))
  smp_amt = 6
  out_u_df = user_df.sample(n=min(len(user_df),smp_amt), random_state=42)
  hal_u_df = user_df.sample(n=min(len(user_df),int(smp_amt/2)),random_state=1)
  hal_a_df = api_df.sample(n=min(len(api_df),int(smp_amt/2)),random_state=2)
  hal_e_df = edit_df.sample(n=min(len(edit_df),int(smp_amt/2)),random_state=3)
  thd_u_df = user_df.sample(n=min(len(user_df),int(smp_amt/3)),random_state=6)
  thd_a_df = api_df.sample(n=min(len(api_df),int(smp_amt/3)),random_state=5)
  thd_e_df = edit_df.sample(n=min(len(edit_df),int(smp_amt/3)),random_state=4)
  out_ua_df = pd.concat([hal_u_df, hal_a_df], ignore_index=True)
  out_ue_df = pd.concat([hal_u_df, hal_e_df], ignore_index=True)
  out_uae_df = pd.concat([thd_u_df, thd_a_df, thd_e_df], ignore_index=True)
  user_text = combine_text(out_u_df)
  ua_text = combine_text(out_ua_df)
  ue_text = combine_text(out_ue_df)
  uae_text = combine_text(out_uae_df)
  return user_text, ua_text, ue_text, uae_text, all_text

def combine_text(temp_df):
  collect_lines = []
  for idx in temp_df.index:
    curr_line = temp_df.loc[idx].copy()
    sent_id = curr_line['sentence_id']
    sent_text = curr_line['sentence_text']
    text_type = curr_line['sentence_author']
    if sent_text !=  'NO_DATA':
      collect_lines.append((sent_id, sent_text, text_type))
  collect_lines = sorted(collect_lines, key = lambda x: x[0])
  _, just_the_text, text_types = zip(*collect_lines)
  session_text = ' '.join(just_the_text)
  return [session_text, list(text_types)]

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
the_df = generate_session_data(the_df, SEED=RANDOM)

#create integer IDs for each author ID
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
X_save_train.to_csv('%s/coauthor_session_train_for_auth_ID.csv'%(data_dir),index=False)
X_save_test.to_csv('%s/coauthor_session_test_for_auth_ID.csv'%(data_dir),index=False)

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


