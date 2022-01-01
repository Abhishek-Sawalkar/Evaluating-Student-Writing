import pandas as pd 
import numpy as np
import os
from tqdm import tqdm

import torch
from transformers import RobertaTokenizerFast, RobertaForTokenClassification
from tokenized_dataset import tokenized_dataset
from config import config
from dataset import Std_Dataset

class inference_class:
    def __init__(self):
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

    def inference(self, sentence, tokenizer, model):

        # std_data = Std_Dataset()

        inputs = tokenizer(sentence,
    #                         is_split_into_words=True, 
                            return_offsets_mapping=True, 
                            padding='max_length', 
                            truncation=True, 
                            max_length=config['max_length'],
                            return_tensors="pt")

        # move to gpu
        ids = inputs["input_ids"].to(config['device'])
        mask = inputs["attention_mask"].to(config['device'])
        # forward pass
        outputs = model(ids, attention_mask=mask, return_dict=False)
        logits = outputs[0]

        active_logits = logits.view(-1, model.num_labels) # shape (batch_size * seq_len, num_labels)
        flattened_predictions = torch.argmax(active_logits, axis=1) # shape (batch_size*seq_len,) - predictions at the token level

        tokens = tokenizer.convert_ids_to_tokens(ids.squeeze().tolist())
        token_predictions = [self.ids_to_labels[i] for i in flattened_predictions.cpu().numpy()]
        wp_preds = list(zip(tokens, token_predictions)) # list of tuples. Each tuple = (wordpiece, prediction)

        prediction = []
        out_str = []
        off_list = inputs["offset_mapping"].squeeze().tolist()
        for idx, mapping in enumerate(off_list):
    #         print(mapping, token_pred[1], token_pred[0],"####")

    #         only predictions on first word pieces are important
            if mapping[0] != 0 and mapping[0] != off_list[idx-1][1]:
    #             print(mapping, token_pred[1], token_pred[0])
                prediction.append(wp_preds[idx][1])
                out_str.append(wp_preds[idx][0])
            else:
                if idx == 1:
                    prediction.append(wp_preds[idx][1])
                    out_str.append(wp_preds[idx][0])
                continue
        return prediction, out_str


    def run_inference(self):

        # std_data = Std_Dataset()

        model1 = RobertaForTokenClassification.from_pretrained(config['model_name'], num_labels=len(self.output_labels))
        model1.load_state_dict(torch.load('model.bin'))

        model1.to(config['device'])

        model1.eval()

        y_pred = []

        
        test_texts = self.get_test_text()

        tokenizer = RobertaTokenizerFast.from_pretrained(config['model_name'])

        for i, t in enumerate(test_texts['text'].tolist()):
            
            o,o_t = self.inference(t, tokenizer, model1)
            y_pred.append(o)
            # l = train_text_df['entities'][i]

        final_preds = []
        # import pdb
        for i in tqdm(range(len(test_texts))):
            idx = test_texts.id.values[i]

            pred = [x.replace('B-','').replace('I-','') for x in y_pred[i]]
        #     print(pred)
            preds = []
            j = 0
            while j < len(pred):
                cls = pred[j]
                if cls == 'O':
                    j += 1
                end = j + 1
                while end < len(pred) and pred[end] == cls:
                    end += 1
                    
                if cls != 'O' and cls != '' and end - j > 10:
                    final_preds.append((idx, cls, ' '.join(map(str, list(range(j, end))))))
                
                j = end
                
        print(final_preds[1])

        test_df = pd.read_csv('../kaggle/input/feedback-prize-2021/sample_submission.csv')

        sub = pd.DataFrame(final_preds)
        sub.columns = test_df.columns
        print(sub.head())
        sub.to_csv("submission.csv", index=False)

if __name__ == '__main__':
    infer = inference_class()
    infer.run_inference()
