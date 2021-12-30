import numpy as np
import pandas as pd
import os
from tqdm import tqdm
        
class Std_Dataset:
    def __init__(self):
        self.train_text_df = pd.DataFrame()
        self.test_text_df = pd.DataFrame()
        self.labels_to_ids = {}
        self.ids_to_labels = {}
        # Labels
        self.output_labels = ['O', 'B-Lead', 'I-Lead', 'B-Position', 'I-Position', 'B-Claim', 'I-Claim', 'B-Counterclaim', 'I-Counterclaim', 
                'B-Rebuttal', 'I-Rebuttal', 'B-Evidence', 'I-Evidence', 'B-Concluding Statement', 'I-Concluding Statement']

        self.labels_to_ids = {v:k for k,v in enumerate(self.output_labels)}
        self.ids_to_labels = {k:v for k,v in enumerate(self.output_labels)}

    def get_test_text(self):
        test_names, test_texts = [], []
        for f in tqdm(list(os.listdir('../kaggle/input/feedback-prize-2021/test'))):
            test_names.append(f.replace('.txt', ''))
            test_texts.append(open('../kaggle/input/feedback-prize-2021/test/' + f, 'r').read())
        test_texts = pd.DataFrame({'id': test_names, 'text': test_texts})

        return test_texts

    def get_train_text(self):
        test_names, train_texts = [], []
        for f in tqdm(list(os.listdir('../kaggle/input/feedback-prize-2021/train'))):
            test_names.append(f.replace('.txt', ''))
            train_texts.append(open('../kaggle/input/feedback-prize-2021/train/' + f, 'r').read())
        train_text_df = pd.DataFrame({'id': test_names, 'text': train_texts})

        return train_text_df

    def adding_labels(self, train_text_df, train_df):
        all_entities = []
        for i in tqdm(train_text_df.iterrows()):
            total = i[1]['text'].split().__len__()
            start = -1
            entities = []
            for j in train_df[train_df['id'] == i[1]['id']].iterrows():
                discourse = j[1]['discourse_type']
                list_ix = j[1]['predictionstring'].split()
                ent = [f"I-{discourse}" for ix in list_ix]
                ent[0] = f"B-{discourse}"
                ds = int(list_ix[0])
                de = int(list_ix[-1])
                if start < ds-1:
                    ent_add = ['O' for ix in range(int(ds-1-start))]
                    ent = ent_add + ent
                entities.extend(ent)
                start = de
            if len(entities) < total:
                ent_add = ["O" for ix in range(total-len(entities))]
                entities += ent_add
            else:
                entities = entities[:total]
            all_entities.append(entities)
        
        train_text_df['entities'] = all_entities

        return train_text_df


    def dataset(self):
        
        train_df = pd.read_csv('../kaggle/input/feedback-prize-2021/train.csv')

        self.test_texts = pd.read_csv('test_text.csv')
        # test_texts = self.get_test_text()
        # test_texts.to_csv('test_text.csv')
        
        # train_text_df = pd.read_csv('train_text_df.csv')
        train_text_df = self.get_train_text()

        self.train_text_df = self.adding_labels(train_text_df, train_df)
        print(train_text_df.head())
        train_text_df.to_csv('train_text_df.csv')


        # Labels
        output_labels = ['O', 'B-Lead', 'I-Lead', 'B-Position', 'I-Position', 'B-Claim', 'I-Claim', 'B-Counterclaim', 'I-Counterclaim', 
                'B-Rebuttal', 'I-Rebuttal', 'B-Evidence', 'I-Evidence', 'B-Concluding Statement', 'I-Concluding Statement']

        self.labels_to_ids = {v:k for k,v in enumerate(output_labels)}
        self.ids_to_labels = {k:v for k,v in enumerate(output_labels)}


if __name__=="__main__":
    data = Std_Dataset()
    data.dataset()

