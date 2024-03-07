#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import pathlib
import random
import pandas as pd
from sklearn.model_selection import GroupShuffleSplit

def prep_detect_HGE_segment(a_df, num_lbl, SEED=42):
  random.seed(SEED)
  a_df = a_df.loc[a_df['sentence_text'] != 'NO_DATA'].copy()
  user_df = a_df.loc[a_df['sentence_author'] == 'user'].copy()
  user_df['label'] = 0
  api_df = a_df.loc[a_df['sentence_author'] == 'api'].copy()
  api_df['label'] = 1
  edit_df = a_df.loc[a_df['sentence_author'] == 'user_and_api'].copy()
  edit_df['label'] = 2
  if num_lbl == 2:
    bal_amt = min(len(user_df),len(api_df))
    sub_user_df = user_df.sample(n=bal_amt,random_state=int(SEED))
    sub_api_df = api_df.sample(n=bal_amt,random_state=int(SEED))
    sub_edit_df = pd.DataFrame()
  elif num_lbl == 3:
    bal_amt = min(len(user_df),len(api_df),len(edit_df))
    sub_user_df = user_df.sample(n=bal_amt,random_state=int(SEED))
    sub_api_df = api_df.sample(n=bal_amt,random_state=int(SEED))
    sub_edit_df = edit_df.sample(n=bal_amt,random_state=int(SEED))
  out_df = pd.concat([sub_user_df, sub_api_df, sub_edit_df],ignore_index=True)
  return out_df

def prep_detect_HGE_session(a_df, num_lbl, SEED=42):
  list_of_sessions = list(set(a_df['session_id']))
  session_out_df = pd.DataFrame()
  count = 0
  for sesh in list_of_sessions:
    sub_df = a_df.loc[a_df['session_id'] == sesh].copy()
    auth = list(sub_df['author_id'])[0]
    prompt = list(sub_df['prompt_id'])[0]
    out_u_df = pd.DataFrame.from_dict([{'session_id':sesh,'author_id':auth,'prompt_id':prompt,'sentence_author':'user'}])
    out_a_df = pd.DataFrame.from_dict([{'session_id':sesh,'author_id':auth,'prompt_id':prompt,'sentence_author':'api'}])
    out_e_df = pd.DataFrame.from_dict([{'session_id':sesh,'author_id':auth,'prompt_id':prompt,'sentence_author':'user_and_api'}])
    user_text, api_text, edit_text, _ = reconstruct_session(sub_df)
    out_u_df['sentence_text'] = user_text
    out_a_df['sentence_text'] = api_text
    out_e_df['sentence_text'] = edit_text
    combo_df = pd.concat([out_u_df, out_a_df, out_e_df],ignore_index=True)
    session_out_df = session_out_df._append(combo_df,ignore_index=True)
  session_out_df = prep_detect_HGE_segment(session_out_df, num_lbl, SEED=SEED)
  return session_out_df

#collects user/api/edited segments to generate session text in order
#can filter out any of the above categories
def reconstruct_session(a_df):
  user_df = a_df.loc[a_df['sentence_author'] == 'user'].copy()
  api_df = a_df.loc[a_df['sentence_author'] == 'api'].copy()
  edit_df = a_df.loc[a_df['sentence_author'] == 'user_and_api'].copy()
  all_text = combine_text(a_df)
  smp_amt = 6
  out_u_df = user_df.sample(n=min(len(user_df),smp_amt), random_state=42)
  out_a_df = api_df.sample(n=min(len(api_df),smp_amt), random_state=42)
  out_e_df = edit_df.sample(n=min(len(edit_df),smp_amt), random_state=42)
  user_text = combine_text(out_u_df)
  api_text = combine_text(out_a_df)
  edit_text = combine_text(out_e_df)
  return user_text, api_text, edit_text, all_text

def generate_sequence_dataframes(train_df, num_trials=10, seed=6):
  random.seed(seed)
  df_list = []
  seq_of_trials = random.sample(list(range(0,666)),k=num_trials)
  labels_for_choice = []
  set_of_labels = set(train_df['label'])
  user_df = train_df.loc[train_df['label'] == 0].copy()
  api_df = train_df.loc[train_df['label'] == 1].copy()
  edit_df = train_df.loc[train_df['label'] == 2].copy()
  for idx in range(len(seq_of_trials)):
    if len(set_of_labels) == 2:
      min_len = min(len(user_df),len(api_df))
      samp_amt = int(0.9 * min_len)
      sub_u_df = user_df.sample(n=samp_amt, random_state=int(idx))
      sub_a_df = api_df.sample(n=samp_amt, random_state=int(idx))
      sub_e_df = pd.DataFrame()
    elif len(set_of_labels) == 3:
      min_len = min(len(user_df),len(api_df),len(edit_df))
      samp_amt = int(0.9 * min_len)
      sub_u_df = user_df.sample(n=samp_amt, random_state=int(idx))
      sub_a_df = api_df.sample(n=samp_amt, random_state=int(idx))
      sub_e_df = edit_df.sample(n=samp_amt, random_state=int(idx))

    sub_train_df = pd.concat([sub_u_df, sub_a_df, sub_e_df],ignore_index=True)
    df_list.append(sub_train_df)
  return df_list

def combine_text(temp_df):
  collect_lines = []
  for idx in temp_df.index:
    curr_line = temp_df.loc[idx].copy()
    sent_id = curr_line['sentence_id']
    sent_text = curr_line['sentence_text']
    text_type = curr_line['sentence_author']
    if sent_text !=  'NO_DATA':
      collect_lines.append((sent_id, sent_text, text_type))
  if len(collect_lines) == 0:
    session_text = 'NO_DATA'
  else:
    collect_lines = sorted(collect_lines, key = lambda x: x[0])
    _, just_the_text, text_types = zip(*collect_lines)
    session_text = ' '.join(just_the_text)
  return session_text

###############################################################
#Edit parameters here
################################################################
coauthor_data = pd.read_csv('CoAuthor_Data_segment.csv')
data_dir='EDIT for saved data directory'

#reproducibility seed
RANDOM=42

#number of trials for train and test
num_trials=20

#################################################################

for granularity in ['session', 'segment']:
  #2 labels is human only vs. AI only
  #3 labels is human only vs. AI only vs. AI edited by human  
  for num_lbls in [2,3]:
    if num_lbls == 2:
      file_pfx = 'bin_detect'
    elif num_lbls == 3:
      file_pfx = 'tri_detect'
    if granularity == 'session':
      the_df = prep_detect_HGE_session(coauthor_data, num_lbls, SEED=RANDOM)
    elif granularity == 'segment':
      the_df = prep_detect_HGE_segment(coauthor_data, num_lbls, SEED=RANDOM)
    gs = GroupShuffleSplit(n_splits=1, test_size=.2, random_state=int(42))
    train_idx, test_idx = next(gs.split(the_df,the_df['label'],groups=the_df['prompt_id']))

    #intermediate save of filtered data
    X_save_train = the_df.loc[train_idx].copy()
    X_save_test = the_df.loc[test_idx].copy()
    save_data_out = '%s/HvAI_detect'%(data_dir)
    pathlib.Path(save_data_out).mkdir(parents=True, exist_ok=True)
    X_save_train.to_csv('%s/%s_%s_train.csv'%(save_data_out,file_pfx,granularity),index=False)
    X_save_test.to_csv('%s/%s_%s_test.csv'%(save_data_out,file_pfx,granularity),index=False)

    #save separate train data for each trial
    list_of_usable_df = generate_sequence_dataframes(X_save_train, num_trials=num_trials, seed=RANDOM)
    for trial_num in range(len(list_of_usable_df)):
      curr_df = list_of_usable_df[trial_num]
      curr_df.to_csv('%s/%s_%s_train_trial_%d.csv'%(save_data_out, file_pfx, granularity, trial_num),index=False)

