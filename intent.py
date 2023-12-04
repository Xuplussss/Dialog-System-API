#from pprint import pprint
import argparse
from importlib import import_module
from transformers import pipeline
from config import model_dict
import torch
import os
from opencc import OpenCC


# def set_args():
#     parser = argparse.ArgumentParser()
#     parser.add_argument('--model', default='bert', type=str, required=False, help='model')
#     parser.add_argument('--pretrain_dir', default='./models/xinwen/bert/', type=str, required=False, help='result')
#     args = parser.parse_args()
#     return args


def eval_object(object_):
    if '.' in object_:
        module_, class_ = object_.rsplit('.', 1)
        module_ = import_module(module_)
        return getattr(module_, class_)
    else: 
        module_ = import_module(object_)
        return module_


class Inferenve:
    def get_model_tokenizer(self):
        ClassifyClass = eval_object(model_dict['bert'][1])
        TokenizerClass = eval_object(model_dict['bert'][0])
        ConfigClass = eval_object(model_dict['bert'][2])
        model = ClassifyClass.from_pretrained('./models/xinwen/bert/')
        tokenizer = TokenizerClass.from_pretrained('./models/xinwen/bert/')
        config = ConfigClass.from_pretrained('./models/xinwen/bert/')
        return model, tokenizer, config

    def __init__(self):
        model, tokenizer, config = self.get_model_tokenizer()
        self.classifier = pipeline("sentiment-analysis", model=model, tokenizer=tokenizer, config=config,top_k=None)

    def get_ret(self, sentences):
        r = self.classifier(sentences)
        return r


infer = Inferenve()
cc = OpenCC('tw2s')

def intent_detection(ss):
    a = infer.get_ret(cc.convert(str(ss)))
    return a[0][0]['label']
#print(a[0][0]['label'])   