Information about CoAuthor data

After running the notebook [Prepare CoAuthor data from raw to plain text.ipynb](https://github.com/AARichburg/Human-AI_Authorship_Analysis/blob/main/Prepare_CoAuthor_data_from_raw_to_plain_text.ipynb) you will have
access to the files CoAuthor_Data_session.csv and CoAuthor_Data_segment.csv.  These CSV files
contain the session/segment level splits of the [CoAuthor Data](https://coauthor.stanford.edu/).
There are the following seven fields:

1) sentence_id: the number sentence for the session.  Note some sessions do not 
start the sentence_id at 0; the prompt texts were removed (can be found at the a
bove link).
2) sentence_author: the "author" of the text; either user (human), api (GPT3) or
 user_and_api (GPT3 edited by a human)
3) sentence_text: the text for a given session/segment; untokenized
4) session_id: the ID for the given session
5) author_id: the ID of the user/human responding to the prompt
6) prompt_id: the ID of the session prompt
7) length: the length of the tokenized text

Example use:
```
import pandas as pd
coauthor_data = pd.read_csv('CoAuthor_Data_segment.csv')

#filter by session
filtered_session_df = coauthor_data.loc[coauthor_data['session_id'] == SESSION_ID]

#filter by author ID then extract human and ai only texts
filtered_author_df = coauthor_data.loc[coauthor_data['author_id'] == AUTHOR_ID]
user_samples = filtered_author_df.loc[filtered_author_df['sentence_author'] == 'user']
api_samples = filtered_author_df.loc[filtered_author_df['sentence_author'] == 'api']
combine_user_api_samples = pd.concat([user_samples, api_samples], ignore_index=True)
```
Once you have extracted the text data into CoAuthor_Data_session.csv and CoAuthor_Data_segment.csv, you can run train_test_split_human_v_AI.py and train_test_split_author_ID.py to create the train/test split across trials as used in the paper.

The code to train and evaluate the character n-gram model are hosted at the repo for the PAN2022 shared task [here](https://github.com/pan-webis-de/pan-code/tree/master/clef22/authorship-verification).
