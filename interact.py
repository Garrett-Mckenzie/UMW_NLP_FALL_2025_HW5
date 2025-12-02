import pandas as pd
import numpy as np
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer , AutoModelForCausalLM
import sys
from tqdm import tqdm

def load():
    model = AutoModelForCausalLM.from_pretrained("./model")
    tokenizer = AutoTokenizer.from_pretrained("distilgpt2")
    return (model,tokenizer)

def getOutput(x,model,tokenizer,maxWords = 100 , distributionSize = 1000 , temperature= 1):
    input_ids = tokenizer(x,return_tensors='pt')['input_ids'].squeeze(0)

    with torch.no_grad():
        for i in range(maxWords):
            #raw logits
            output = model(input_ids=input_ids.unsqueeze(0))
            next_logits = output.logits[0,-1,:]

            #temperature and get topk
            adjusted_logits = next_logits * (1/temperature) 
            next_logits, next_tokens = torch.topk(adjusted_logits,distributionSize)

            #get out probs and indx
            probs = torch.softmax(adjusted_logits,dim = -1)
            chosen_indx = torch.multinomial(probs , num_samples = 1)

            #get out the next token
            token = next_tokens[chosen_indx]
            input_ids = torch.cat([input_ids,token])
            print(tokenizer.decode(token),end = "")

def chat():
    model,tokenizer = load()
    print("!! Welcome to the fine tune chat room !!\n")
    print("Below you can chat with the bot, whenever you are done simply type done to terminate the program.\n")
    while True:
        x = input("Type something to the model: ")

        if x == "done":
            print("!!! Later Nerd !!!")
            return

        response = getOutput(x,model,tokenizer)
        print("The model said: " + response)

def main():
    chat()

if __name__=="__main__":
    main()
