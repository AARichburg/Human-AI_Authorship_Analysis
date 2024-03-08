#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import pathlib
import random
import pandas as pd
from sklearn.model_selection import train_test_split

def generate_verification_data(a_df, num_seeds=60, SEED=893):
  random.seed(SEED)
  out_df = pd.DataFrame(columns=['session_id_pair','author_id_pair','prompt_id_pair','user_text_pair','user_api_text_pair','user_edit_text_pair','user_api_edit_text_pair','label'])
  set_of_authors = sorted(list(set(a_df['author_id'])))
  text_opts = ['user_text','user_api_text','user_edit_text','user_api_edit_text']
  for auth in set_of_authors:
    sub_df_orig = a_df.loc[a_df['author_id'] == auth].copy().reset_index(drop=True)
    seed_bag = random.sample(list(range(0,666)),k=num_seeds)
    for semilla in seed_bag:
      sub_df_shuf = sub_df_orig.sample(frac=1, random_state=semilla).reset_index(drop=True)
      conv_sub_df = a_df.loc[a_df['author_id'] != auth].copy().sample(n=len(sub_df_orig.index),random_state=semilla).reset_index(drop=True)
      for idx in range(len(sub_df_orig.index)):
        orig_row = sub_df_orig.loc[idx].copy()
        shuf_row = sub_df_shuf.loc[idx].copy()
        conv_row = conv_sub_df.loc[idx].copy()
        o_sess, o_auth, o_prmpt = quick_fetch_id_info(orig_row)
        s_sess, s_auth, s_prmpt = quick_fetch_id_info(shuf_row)
        c_sess, c_auth, c_prmpt = quick_fetch_id_info(conv_row)

        os_sess_pair = (o_sess, s_sess)
        os_auth_pair = (o_auth, s_auth)
        os_prmpt_pair = (o_prmpt, s_prmpt)
        oc_sess_pair = (o_sess, c_sess)
        oc_auth_pair = (o_auth, c_auth)
        oc_prmpt_pair = (o_prmpt, c_prmpt)

        u_tex_o = orig_row['user_text']
        u_tex_s = shuf_row['user_text']
        u_tex_c = conv_row['user_text']
        ua_tex_o = orig_row['user_api_text']
        ua_tex_s = shuf_row['user_api_text']
        ua_tex_c = conv_row['user_api_text']
        ue_tex_o = orig_row['user_edit_text']
        ue_tex_s = shuf_row['user_edit_text']
        ue_tex_c = conv_row['user_edit_text']
        uae_tex_o = orig_row['user_api_edit_text']
        uae_tex_s = shuf_row['user_api_edit_text']
        uae_tex_c = conv_row['user_api_edit_text']
        #os positive pairs, oc negative pairs
        u_os_text_pair = (u_tex_o, u_tex_s)
        u_oc_text_pair = (u_tex_o, u_tex_c)
        ua_os_text_pair = (ua_tex_o, ua_tex_s)
        ua_oc_text_pair = (ua_tex_o, ua_tex_c)
        ue_os_text_pair = (ue_tex_o, ue_tex_s)
        ue_oc_text_pair = (ue_tex_o, ue_tex_c)
        uae_os_text_pair = (uae_tex_o, uae_tex_s)
        uae_oc_text_pair = (uae_tex_o, uae_tex_c)
        os_out = {'session_id_pair':os_sess_pair, 'author_id_pair':os_auth_pair, 'prompt_id_pair':os_prmpt_pair, 'user_text_pair':u_os_text_pair, 'user_api_text_pair':ua_os_text_pair, 'user_edit_text_pair':ue_os_text_pair, 'user_api_edit_text_pair':uae_os_text_pair, 'label':1}
        oc_out = {'session_id_pair':oc_sess_pair, 'author_id_pair':oc_auth_pair, 'prompt_id_pair':oc_prmpt_pair, 'user_text_pair':u_oc_text_pair, 'user_api_text_pair':ua_oc_text_pair, 'user_edit_text_pair':ue_oc_text_pair, 'user_api_edit_text_pair':uae_oc_text_pair, 'label':0}
        out_df = out_df._append(os_out, ignore_index=True)
        out_df = out_df._append(oc_out, ignore_index=True)

    out_df.drop_duplicates(inplace=True)
    num_pos = out_df.loc[out_df['label'] == 1]
    num_neg = out_df.loc[out_df['label'] == 0]
    min_len = min(len(num_pos.index),len(num_neg.index))
    out_pos = num_pos.sample(n=min_len,random_state=SEED).reset_index(drop=True)
    out_neg = num_neg.sample(n=min_len,random_state=SEED).reset_index(drop=True)
    out_df= pd.concat([out_pos,out_neg],ignore_index=True)
  return out_df

def quick_fetch_id_info(a_df):
  sess_id = a_df['session_id']
  auth_id = a_df['author_id']
  prmpt_id = a_df['prompt_id']
  return sess_id, auth_id, prmpt_id

def generate_sequence_dataframes(train_df, num_trials=10, seed=6):
  random.seed(seed)
  df_list = []
  seq_of_trials = random.sample(list(range(0,666)),k=num_trials)
  pos_exs = train_df.loc[train_df['label'] == 1].copy()
  neg_exs = train_df.loc[train_df['label'] == 0].copy()
  samp_size = int(len(pos_exs) * 0.9)
  for idx in range(len(seq_of_trials)):
    sub_pos_df = pos_exs.sample(n=samp_size, random_state=idx)
    sub_neg_df = neg_exs.sample(n=samp_size, random_state=idx)
    sub_train_df = pd.concat([sub_pos_df, sub_neg_df],ignore_index=True)
    df_list.append(sub_train_df)
  return df_list

#####################################################################
#Edit parameters here
####################################################################
coauth_train = pd.read_csv('coauthor_session_train_for_auth_ID.csv')
coauth_test = pd.read_csv('coauthor_session_test_for_auth_ID.csv')
data_dir='EDIT for saved data directory'

#seed for reproducibility
RANDOM=42

#number of trials for train, dev and test
num_trials=5

#####################################################################

veri_train_df = generate_verification_data(coauth_train,SEED=42)
veri_test_df = generate_verification_data(coauth_test,num_seeds=10,SEED=6)
veri_test_df, veri_dev_df, _, _ = train_test_split(veri_test_df, veri_test_df['label'], test_size=0.5, random_state=RANDOM)

#intermediate save
save_data_out = '%s/auth_verif' % (data_dir)
pathlib.Path(save_data_out).mkdir(parents=True,exist_ok=True)
veri_train_df.to_csv('%s/coauthor_verif_train.csv'%(save_data_out), index=False)
veri_dev_df.to_csv('%s/coauthor_verif_dev.csv'%(save_data_out), index=False)
veri_test_df.to_csv('%s/coauthor_verif_test.csv'%(save_data_out), index=False)

list_of_usable_df = generate_sequence_dataframes(veri_train_df, num_trials=num_trials, seed=6)
for idx in range(len(list_of_usable_df)):
  train_df = list_of_usable_df[idx]
  train_df.to_csv('%s/coauthor_verif_train_%d.csv'%(save_data_out,idx), index=False)


