import numpy as np
import pandas as pd
import os
from tqdm import tqdm

class dataset(Dataset):
    def __init__(self, dataframe, tokenizer, max_len):
        self.len = len(dataframe)
        self.data = dataframe
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __getitem__(self, index):
        # step 1: get the sentence and word labels 
        sentence = self.data.text[index]
        word_labels = self.data.entities[index]

        # step 2: use tokenizer to encode sentence (includes padding/truncation up to max length)
        # BertTokenizerFast provides a handy "return_offsets_mapping" functionality for individual tokens
        encoding = self.tokenizer(sentence,
#                              is_pretokenized=True, 
#                                   is_split_into_words=True,
                             return_offsets_mapping=True, 
                             padding='max_length', 
                             truncation=True, 
                             max_length=self.max_len)
        
        # step 3: create token labels only for first word pieces of each tokenized word
#         pdb.set_trace()
        labels = [labels_to_ids[label] for label in word_labels] 
        # code based on https://huggingface.co/transformers/custom_datasets.html#tok-ner
        # create an empty array of -100 of length max_length
        encoded_labels = np.ones(len(encoding["offset_mapping"]), dtype=int) * -100
#         print(len(sentence), len(labels))
        # set only labels whose first offset position is 0 and the second is not 0
        i = 0
        for idx, mapping in enumerate(encoding["offset_mapping"]):
#             print(idx)
            if mapping[0] != 0 and mapping[0] != encoding['offset_mapping'][idx-1][1]:

                try:
                    encoded_labels[idx] = labels[i]
                except:
                    pass
                i += 1
            else:
                if idx==1:
    #                 print(idx)
                    encoded_labels[idx] = labels[i]
                    i += 1
        # step 4: turn everything into PyTorch tensors
        item = {key: torch.as_tensor(val) for key, val in encoding.items()} # encodeing.items -> input_ids, attention_mask, offset_mapping
        item['labels'] = torch.as_tensor(encoded_labels) # adding labels
        
        return item

    def __len__(self):
        return self.len
        

def get_test_text():
    test_names, test_texts = [], []
    for f in tqdm(list(os.listdir('../kaggle/input/feedback-prize-2021/test'))):
        test_names.append(f.replace('.txt', ''))
        test_texts.append(open('../kaggle/input/feedback-prize-2021/test/' + f, 'r').read())
    test_texts = pd.DataFrame({'id': test_names, 'text': test_texts})

    return test_texts

def get_train_text():
    test_names, train_texts = [], []
    for f in tqdm(list(os.listdir('../kaggle/input/feedback-prize-2021/train'))):
        test_names.append(f.replace('.txt', ''))
        train_texts.append(open('../kaggle/input/feedback-prize-2021/train/' + f, 'r').read())
    train_text_df = pd.DataFrame({'id': test_names, 'text': train_texts})

    return train_text_df

def adding_labels(train_text_df, train_df):
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


def dataset():
    
    train_df = pd.read_csv('../kaggle/input/feedback-prize-2021/train.csv')

    test_texts = get_test_text()
    # print(test_texts)
    train_text_df = get_train_text()

    train_text_df = adding_labels(train_text_df, train_df)
    print(train_text_df.head())

    # Labels
    output_labels = ['O', 'B-Lead', 'I-Lead', 'B-Position', 'I-Position', 'B-Claim', 'I-Claim', 'B-Counterclaim', 'I-Counterclaim', 
            'B-Rebuttal', 'I-Rebuttal', 'B-Evidence', 'I-Evidence', 'B-Concluding Statement', 'I-Concluding Statement']

    labels_to_ids = {v:k for k,v in enumerate(output_labels)}
    ids_to_labels = {k:v for k,v in enumerate(output_labels)}


if __name__=="__main__":
    dataset()

