import numpy as np
import pandas as pd
from tokenized_dataset import tokenized_dataset, gen_token
from config import config
from dataset import Std_Dataset

import torch
from torch import cuda
from sklearn.metrics import accuracy_score
from torch.utils.data import Dataset, DataLoader


# Defining the training function on the 80% of the dataset for tuning the bert model
class training:
    def __init__(self):
        print("TRAINING.py RUNNING")


    def train(self, epoch, model, training_loader):
        
        std_data = Std_Dataset()

        device = config['device']
        optimizer = torch.optim.Adam(params=model.parameters(), lr=config['learning_rate'])

        tr_loss, tr_accuracy = 0, 0
        nb_tr_examples, nb_tr_steps = 0, 0
        tr_preds, tr_labels = [], []
        # put model in training mode
        model.train()
        
        for idx, batch in enumerate(training_loader):
            
            ids = batch['input_ids'].to(device, dtype = torch.long)
            mask = batch['attention_mask'].to(device, dtype = torch.long)
            labels = batch['labels'].to(device, dtype = torch.long)

            loss, tr_logits = model(input_ids=ids, attention_mask=mask, labels=labels,
                                return_dict=False)
            tr_loss += loss.item()

            nb_tr_steps += 1
            nb_tr_examples += labels.size(0)
            
            if idx % 100==0:
                loss_step = tr_loss/nb_tr_steps
                print(f"Training loss per 100 training steps: {loss_step}")
            
            # compute training accuracy
            flattened_targets = labels.view(-1) # shape (batch_size * seq_len,)
            active_logits = tr_logits.view(-1, model.num_labels) # shape (batch_size * seq_len, num_labels)
            flattened_predictions = torch.argmax(active_logits, axis=1) # shape (batch_size * seq_len,)
            
            # only compute accuracy at active labels
            active_accuracy = labels.view(-1) != -100 # shape (batch_size, seq_len)
            #active_labels = torch.where(active_accuracy, labels.view(-1), torch.tensor(-100).type_as(labels))
            
            labels = torch.masked_select(flattened_targets, active_accuracy)
            predictions = torch.masked_select(flattened_predictions, active_accuracy)
            
            tr_labels.extend(labels)
            tr_preds.extend(predictions)

            tmp_tr_accuracy = accuracy_score(labels.cpu().numpy(), predictions.cpu().numpy())
            tr_accuracy += tmp_tr_accuracy
        
            # gradient clipping
            torch.nn.utils.clip_grad_norm_(
                parameters=model.parameters(), max_norm=config['max_grad_norm']
            )
            
            # backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        epoch_loss = tr_loss / nb_tr_steps
        tr_accuracy = tr_accuracy / nb_tr_steps
        print(f"Training loss epoch: {epoch_loss}")
        print(f"Training accuracy epoch: {tr_accuracy}")

    def run_training(self):
        training_set, testing_set, test_texts_set, model = gen_token()

        # Params for dividing the dataset into batches
        train_params = {'batch_size': config['train_batch_size'],
                    'shuffle': True,
                    'num_workers': 1,
                    'pin_memory':True
                    }

        test_params = {'batch_size': config['valid_batch_size'],
                        'shuffle': True,
                        'num_workers': 1,
                        'pin_memory':True
                        }

        training_loader = DataLoader(training_set, **train_params)
        testing_loader = DataLoader(testing_set, **test_params)
        test_texts_loader = DataLoader(test_texts_set, **test_params)

        device = config['device'] # Device
        optimizer = torch.optim.Adam(params=model.parameters(), lr=config['learning_rate'])

        for epoch in range(config['epochs']):
            print(f"Training epoch: {epoch + 1}")
            self.train(epoch, model, training_loader)


if __name__ == "__main__":
    train = training()
    train.run_training()