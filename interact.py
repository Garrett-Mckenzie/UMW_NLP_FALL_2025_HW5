import pandas as pd
import numpy as np
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer , AutoModelForCausalLM
import sys
from tqdm import tqdm

def load(modelPath = "./model"):
    model = AutoModelForCausalLM.from_pretrained(modelPath)
    tokenizer = AutoTokenizer.from_pretrained("distilgpt2")
    return (model,tokenizer)

def getOutput(x,model,tokenizer,maxWords = 100 , distributionSize = 1000 , temperature= 10):
    input_ids = tokenizer(x,return_tensors='pt')['input_ids'].squeeze(0)
    return_string = ""

    with torch.no_grad():
        for i in range(maxWords):
            #raw logits
            output = model(input_ids=input_ids.unsqueeze(0))
            next_logits = output.logits[0,-1,:]

            #temperature and get topk 
            vals , ids = torch.topk((next_logits / temperature),distributionSize)

            #get out probs and indx
            probs = torch.softmax(vals,dim = -1)
            chosen_index = torch.multinomial(probs , num_samples = 1)
            chosen_id = ids[chosen_index]

            input_ids = torch.cat([input_ids,chosen_id])
            return_string += tokenizer.decode(chosen_id)

    return return_string

def chat(maxWords, distributionSize, temperature,modelPath=None):
    model = None
    tokenizer = None

    if modelPath == None:
        model,tokenizer = load()
    else:
        model,tokenizer = load(modelPath)
    print("!! Welcome to the fine tune chat room !!\n")
    print("Below you can chat with the bot, whenever you are done simply type done to terminate the program.\n")
    while True:
        x = input("\nType something to the model:\n")

        if x == "done":
            print("\n!!! Later Nerd !!!\n")
            return

        response = getOutput(x,model,tokenizer,maxWords=maxWords,distributionSize=distributionSize,temperature=temperature)
        print("\nThe model said:\n" + response)

def main():

    temp = 10
    if "--temp" in sys.argv:
        try:
            index = sys.argv.index("temperature")
            temp = float(sys.argv[index + 1])
            
        except:
            print("bad temp arg")
            return
    
    maxWords = 100
    if "--maxWords" in sys.argv:
        try:
            index = sys.argv.index("--maxWords")
            maxWords = float(sys.argv[index + 1])
        except:
            print("bad maxWords arg")
            return

    distSize = 1000
    if "--distSize" in sys.argv:
        try:
            index = sys.argv.index("--distSize")
            distSize = float(sys.argv[index + 1])
        except:
            print("bad distSize arg")
            return

    modelPath = None
    if "--modelPath" in sys.argv:
        try:
            index = sys.argv.index("--modelPath")
            modelPath = sys.argv[index + 1]
        except:
            print("bad model path")
            return

    chat(maxWords,distSize,temp,modelPath)

if __name__=="__main__":
    main()
