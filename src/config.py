from torch import cuda

config = {'model_name': '../kaggle/input/roberta-base/',
         'max_length': 512,
         'train_batch_size':8,
         'valid_batch_size':16,
         'epochs':3,
         'learning_rate':1e-05,
         'max_grad_norm':10,
         'device': 'cuda' if cuda.is_available() else 'cpu'}

print(f"cuda: {config['device']}")