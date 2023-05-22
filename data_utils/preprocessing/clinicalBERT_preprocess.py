"""modified from https://github.com/kexinhuang12345/clinicalBERT/blob/master/preprocess.py"""
import os
import re

import numpy as np
import pandas as pd
import tqdm
from pytorch_pretrained_bert.tokenization import BertTokenizer

import _settings
from data_utils.preprocessing.clinicalBERT_dataloader import (
    InputExample, convert_examples_to_features)

DATA_PATH = _settings.MIMIC_PATH
OUTPUT_PATH = _settings.WORKSPACE



def note_remove_diagnoses(
    text,
    patterns=["^\n[\w ]*diagnos[ie]s:(?=.*\n)", "^\n[\w\s]*medicati[\w\s]*:(?=.*\n)", ],
    general_pattern='^\n.*:(?=.*\n)',
     flags=re.IGNORECASE + re.M):
    section_to_remove = set([match.span() for patt in patterns for match in re.finditer(patt, text, flags=flags)])
    #print([re.search(patt, text, flags=flags) for patt in patterns])
    to_dels = []
    removing=None
    for matched in re.finditer(general_pattern, text, flags=flags):
        #print(matched)
        curr_span = matched.span()
        if removing is not None:
            to_del = text[removing[0]:curr_span[0]+1]
            if re.search('\n\s*\n', to_del):
                to_dels.append(to_del)
                #print(f"||\n{to_del}\n||")
                section_to_remove.remove(removing)
                removing = None
        if curr_span in section_to_remove:
            assert removing is None
            removing = curr_span
    for to_del in to_dels:
        text = text.replace(to_del, '\n')
    return text

def note_preprocess1(x):
    y = ' ' if pd.isnull(x) else x
    y = y.replace('\n',' ').replace('\r',' ').strip().lower()
    y=re.sub('\\[(.*?)\\]','',y) #remove de-identified brackets
    y=re.sub('[0-9]+\.','',y) #remove 1.2. since the segmenter segments based on this
    y=re.sub('dr\.','doctor',y)
    y=re.sub('m\.d\.','md',y)
    y=re.sub('admission date:','',y)
    y=re.sub('discharge date:','',y)
    y=re.sub('--|__|==','',y)
    return y

def preprocessing(df_less_n):
    df_less_n['TEXT']=df_less_n['TEXT'].apply(note_remove_diagnoses)
    # only around 5% do not have the sections removed
    df_less_n['TEXT']=df_less_n['TEXT'].apply(note_preprocess1)
    #return df_less_n.reindex(columns=['HADM_ID', 'TEXT'])

    #to get 318 words chunks for readmission tasks
    df_len = len(df_less_n)
    want = []
    for i in tqdm.tqdm(range(df_len)):
        x=df_less_n.TEXT.iloc[i].split()
        hadm_id = df_less_n['HADM_ID'].iloc[i]
        for j in range(0, len(x), 318):
            j_ = min(len(x), j + 318)
            if j_ - j > 10:
                want.append({'TEXT':' '.join(x[j:j_]),'Seq': j//318,'HADM_ID':hadm_id})
    want = pd.DataFrame(want)
    return want

def read_filtered_df():
    df_adm = pd.read_csv(os.path.join(DATA_PATH, 'ADMISSIONS.csv.gz'), compression='gzip')
    df_adm.ADMITTIME = pd.to_datetime(df_adm.ADMITTIME, format = '%Y-%m-%d %H:%M:%S', errors = 'coerce')
    df_adm.DISCHTIME = pd.to_datetime(df_adm.DISCHTIME, format = '%Y-%m-%d %H:%M:%S', errors = 'coerce')
    df_adm.DEATHTIME = pd.to_datetime(df_adm.DEATHTIME, format = '%Y-%m-%d %H:%M:%S', errors = 'coerce')

    df_adm = df_adm.sort_values(['SUBJECT_ID','ADMITTIME'])
    df_adm = df_adm.reset_index(drop = True)
    df_adm['NEXT_ADMITTIME'] = df_adm.groupby('SUBJECT_ID').ADMITTIME.shift(-1)
    df_adm['NEXT_ADMISSION_TYPE'] = df_adm.groupby('SUBJECT_ID').ADMISSION_TYPE.shift(-1)

    rows = df_adm.NEXT_ADMISSION_TYPE == 'ELECTIVE'
    df_adm.loc[rows,'NEXT_ADMITTIME'] = pd.NaT
    df_adm.loc[rows,'NEXT_ADMISSION_TYPE'] = np.NaN

    df_adm = df_adm.sort_values(['SUBJECT_ID','ADMITTIME'])

    #When we filter out the "ELECTIVE", we need to correct the next admit time for these admissions since there might be 'emergency' next admit after "ELECTIVE"
    df_adm[['NEXT_ADMITTIME','NEXT_ADMISSION_TYPE']] = df_adm.groupby(['SUBJECT_ID'])[['NEXT_ADMITTIME','NEXT_ADMISSION_TYPE']].fillna(method = 'bfill')
    df_adm['DAYS_NEXT_ADMIT']=  (df_adm.NEXT_ADMITTIME - df_adm.DISCHTIME).dt.total_seconds()/(24*60*60)
    df_adm['OUTPUT_LABEL'] = (df_adm.DAYS_NEXT_ADMIT < 30).astype('int')
    ### filter out newborn and death
    df_adm = df_adm[df_adm['ADMISSION_TYPE']!='NEWBORN']
    df_adm = df_adm[df_adm.DEATHTIME.isnull()]
    df_adm['DURATION'] = (df_adm['DISCHTIME']-df_adm['ADMITTIME']).dt.total_seconds()/(24*60*60)

    df_notes = pd.read_csv(os.path.join(DATA_PATH, 'NOTEEVENTS.csv.gz'), compression='gzip')
    df_notes = df_notes.sort_values(by=['SUBJECT_ID','HADM_ID','CHARTDATE'])
    df_adm_notes = pd.merge(df_adm[['SUBJECT_ID','HADM_ID','ADMITTIME','DISCHTIME','DAYS_NEXT_ADMIT','NEXT_ADMITTIME','ADMISSION_TYPE','DEATHTIME','OUTPUT_LABEL','DURATION']],
                            df_notes[['SUBJECT_ID','HADM_ID','CHARTDATE','TEXT','CATEGORY']],
                            on = ['SUBJECT_ID','HADM_ID'],
                            how = 'left')

    df_adm_notes['ADMITTIME_C'] = df_adm_notes['ADMITTIME'].apply(lambda x: str(x).split(' ')[0])
    df_adm_notes['ADMITTIME_C'] = pd.to_datetime(df_adm_notes['ADMITTIME_C'], format = '%Y-%m-%d', errors = 'coerce')
    df_adm_notes['CHARTDATE'] = pd.to_datetime(df_adm_notes['CHARTDATE'], format = '%Y-%m-%d', errors = 'coerce')

    ### If Discharge Summary
    df_discharge = df_adm_notes[df_adm_notes['CATEGORY'] == 'Discharge summary']
    # multiple discharge summary for one admission -> after examination -> replicated summary -> replace with the last one
    df_discharge = (df_discharge.groupby(['SUBJECT_ID','HADM_ID']).nth(-1)).reset_index()
    df_discharge = df_discharge[df_discharge['TEXT'].notnull()]

    return df_discharge


def read_bert_features(df:pd.DataFrame, tokenizer=None):
    if tokenizer is None:
        tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    if isinstance(df, str):
        df = pd.read_csv(df)

    examples = []
    for idx, r in df.reindex(columns=['HADM_ID', 'Seq', 'TEXT']).iterrows():
        examples.append(
            InputExample(guid=f"{r['HADM_ID']}-{r['Seq']}", text_a=r['TEXT'], text_b=None, label='0'))

    features = convert_examples_to_features(examples, ["0", "1"], 512, tokenizer)
    res = {'input_ids':[], 'input_mask': [], 'segment_ids': [], 'guid': []}
    for feature in tqdm.tqdm(features):
        for key in res.keys():
            res[key].append(getattr(feature, key))
    meta = pd.DataFrame(np.asarray([_.split("-") for _ in res['guid']]), columns=['HADM_ID', 'Seq'])
    return meta, np.asarray(res['input_ids']), np.asarray(res['segment_ids']), np.asarray(res['input_mask'])

def main():
    cache_path = os.path.join(OUTPUT_PATH, "discharge_new.csv")
    if not os.path.isfile(cache_path):
        preprocessing(read_filtered_df()).to_csv(cache_path, index=False)

if __name__ == '__main__':
    main()

