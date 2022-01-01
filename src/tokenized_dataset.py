import numpy as np
import pandas as pd
import pickle
import _pickle as cPickle
import bz2

from dataset import Std_Dataset
from config import config

import torch
from transformers import RobertaTokenizerFast, RobertaForTokenClassification
from torch.utils.data import Dataset, DataLoader

class tokenized_dataset(Dataset):
  def __init__(self, dataframe, tokenizer, max_len, model, std_data):
        self.len = len(dataframe)
        self.data = pd.DataFrame(dataframe)
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.model = model
        self.std_data = std_data

  def __len__(self):
        return self.len

  def __getitem__(self, index):
        # step 1: get the sentence and word labels 
        sentence = self.data.text[index]
        word_labels = list(self.data.entities[index])
        # print(f'DT:\n {self.data.entities.dtype}')
        
        wl = set(word_labels)
        # print(f"word_labels: {word_labels[0]}")

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
        labels_to_ids = self.std_data.labels_to_ids
        # print(f'labels_to_ids:\n {(labels_to_ids)}')
        labels = [ labels_to_ids[label] for label in word_labels] 
        
        # code based on https://huggingface.co/transformers/custom_datasets.html#tok-ner
        # create an empty array of -100 of length max_length
        encoded_labels = np.ones(len(encoding["offset_mapping"]), dtype=int) * -100
#         print(len(sentence), len(labels))
        # set only labels whose first offset position is 0 and the second is not 0
        i = 0
        for idx, mapping in enumerate(encoding["offset_mapping"]):
#             print(idx)
            if mapping[0] != 0 and mapping[0] != encoding['offset_mapping'][idx-1][1]:
            # overwrite label
#             pdb.set_trace()
#             print(mapping)
#             print(encoded_labels.shape, len(labels), idx, i)
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
        item = {key: torch.as_tensor(val) for key, val in encoding.items()}
        item['labels'] = torch.as_tensor(encoded_labels)
        
        return item


def gen_token():

    std_data = Std_Dataset()
    # Reading Data
    data = bz2.BZ2File('test_texts.pbz2', 'rb')
    test_text = cPickle.load(data)
    test_text = pd.DataFrame(test_text)

    data = bz2.BZ2File('train_text_df.pbz2', 'rb')
    train_text_df = cPickle.load(data)
    train_text_df = pd.DataFrame(train_text_df)

    # Tokenizer and Model

    tokenizer = RobertaTokenizerFast.from_pretrained(config['model_name'])
    model = RobertaForTokenClassification.from_pretrained(config['model_name'], num_labels=len(std_data.output_labels))

    # Dividing the train data into train and test/valid data into 80-20 split.
    data = train_text_df[['text', 'entities']]
    train_size = 0.8
    train_dataset = data.sample(frac=train_size,random_state=200)
    test_dataset = data.drop(train_dataset.index).reset_index(drop=True)
    train_dataset = train_dataset.reset_index(drop=True)

    print("FULL Dataset: {}".format(data.shape))
    print("TRAIN Dataset: {}".format(train_dataset.shape))
    print("TEST Dataset: {}".format(test_dataset.shape))

    # Appying the class
    training_set = tokenized_dataset(train_dataset, tokenizer, config['max_length'], model, std_data)
    testing_set = tokenized_dataset(test_dataset, tokenizer, config['max_length'], model, std_data)

    # Params for dividing the dataset into batches
    # train_params = {'batch_size': config['train_batch_size'],
    #             'shuffle': True,
    #             'num_workers': 1,
    #             'pin_memory':True
    #             }

    # test_params = {'batch_size': config['valid_batch_size'],
    #                 'shuffle': True,
    #                 'num_workers': 1,
    #                 'pin_memory':True
    #                 }

    # training_loader = DataLoader(training_set, **train_params)
    # testing_loader = DataLoader(testing_set, **test_params)

    # For the hidden test data
    test_texts_set = tokenized_dataset(test_text, tokenizer, config['max_length'], model, std_data)
    # test_texts_loader = DataLoader(test_texts_set, **test_params)

    return training_set, testing_set, test_texts_set, model


if __name__=="__main__":
        gen_token()
