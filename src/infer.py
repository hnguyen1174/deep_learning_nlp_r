from transformers import AutoModelForSequenceClassification, AutoConfig, AutoTokenizer, pipeline
import pandas as pd
import numpy as np
import torch
import sys

if __name__ == '__main__':

    symptom_model_dict = {
        'Cough': 'cough_distilbert',
        'Chest Pain': 'chest.pain_bert',
        'Dyspnea': 'dyspnea_bert',
        'Fatique': 'fatique_bert',
        'Nausea': 'nausea_bert'
    }
    text = str(sys.argv[0])

    for symptom, model in symptom_model_dict.items():

        model_name = '../models/{}'.format(model)
        model = AutoModelForSequenceClassification.from_pretrained(model_name)
        tokenizer = AutoTokenizer.from_pretrained(model_name + '/tokenizer')

        input_tensor = tokenizer.encode(text, return_tensors="pt")
        logits = model(input_tensor)[0]
        softmax = torch.nn.Softmax(dim=1)
        probs = softmax(logits)[0][1]
        probs = probs.cpu().detach().numpy()

        print('Probability of {} is {}%.'.format(symptom, round(probs*100, 2)))

