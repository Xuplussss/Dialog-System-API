import argparse
import os
import torch
#from .model_1 import BertModel_1
from utils import eval_object
from config import *
from torch import nn
from transformers import BertPreTrainedModel

class BertModel_1(nn.Module):
    def __init__(self,):
        super(BertModel_1, self).__init__()
        ClassifyClass = eval_object(model_dict['bert'][1]) 
        ClassifyConfig = eval_object(model_dict['bert'][2]) 
        bert_path_or_name = model_dict['bert'][-1] 
        config = ClassifyConfig.from_pretrained(bert_path_or_name, num_labels=33, problem_type='multi_label_classification')
        self.bert = ClassifyClass.from_pretrained(bert_path_or_name, num_labels=33, problem_type='multi_label_classification')

        self.device = torch.device("cuda")
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.regression = nn.Linear(33, config.num_labels)
      
    def forward(self, **input_):
        outputs = self.bert(**input_, return_dict = False)
        pooled_output = self.dropout(outputs[0])
        scores = self.regression(pooled_output)
 
        return scores


def get_model_tokenizer():
    ClassifyClass = eval_object(model_dict['bert'][1])
    TokenizerClass = eval_object(model_dict['bert'][0])
    model = BertModel_1().to('cuda')
    model = model.to('cuda')
    tokenizer = TokenizerClass.from_pretrained('./models/xinwen_multi_label/bert/')
    return model, tokenizer


def set_args():
    parser = argparse.ArgumentParser()
    
    # parser.add_argument('--model', default='bert', type=str, required=False, help='model')
    # parser.add_argument('--problem_type', default='multi_label_classification', type=str, required=False, help='single or multi')
    # parser.add_argument('--dir_name', default='xinwen_multi_label', type=str, required=False, help='train.csv test.csv dev.csv')
    args = parser.parse_args()
    args.model = 'bert'
    args.problem_type = 'multi_label_classification'
    args.dir_name = 'xinwen_multi_label'
    return args

def init(args):
    pretrain_dir = f'./models/{args.dir_name}/{args.model}/'
    args.device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    args.pretrain_dir = pretrain_dir
    args.num_labels = 33


# args = set_args()
# init(args)
model, tokenizer = get_model_tokenizer()
model.load_state_dict(torch.load('./models/HY_scores.pt'))


def scoring(ss):
    data = tokenizer(ss, padding=True, truncation=True, return_tensors='pt')
    tokened_data_dict = {k: v.to('cuda') for k, v in data.items()}
    scores = model(**tokened_data_dict).detach().cpu().numpy()[0] 

    return scores


