import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, TensorDataset, DataLoader
from transformers import AutoTokenizer, AutoModelForCausalLM
import sys
from tqdm import tqdm
from torch.optim import AdamW

def load(filepath):
    corpus = None
    with open(filepath , 'r') as file:
        corpus = file.read()
        corpus = corpus.split("\n")
    for i in tqdm(range(len(corpus))):
        corpus[i] = corpus[i].strip() #removes whitespace
    return corpus

def tokenize(corpus,tokenizer):
    returnMe = np.array([]).astype(int)
    for i in tqdm(range(len(corpus))):
        doc = corpus[i]
        returnMe = np.append(returnMe,tokenizer(doc,return_tensors="np")["input_ids"].squeeze(0))
    return returnMe

def seperate(tokenizedCorpus,blockSize):
    length = len(tokenizedCorpus)
    train = []
    target = []
    pos = 0
    while True:
        if pos + blockSize >= length:
            break
        else:
            x = tokenizedCorpus[pos:pos + blockSize - 1]
            y = tokenizedCorpus[pos + 1:pos + blockSize]
            train.append(x)
            target.append(y)
            pos += blockSize
    return (torch.tensor(np.array(train)),torch.tensor(np.array(target)))

def tune(loader,model,epochs,lr,vocabLength,batchSize,testing=False):
    print("Starting Training")

    optimizer = AdamW(model.parameters(),lr = lr)

    while epochs > 0:
        batchCount = 0
        if testing:
            testingRuns = 2
        for x,y in loader:
            y_hat = model(input_ids=x).logits

            #This is kinda gross but it works
            target = np.zeros((batchSize , len(y[0]) , vocabLength))
            for j in range(target.shape[0]):
                for i in range(target.shape[1]):
                    target[j][i][y[j][i]] += 1
            target = torch.tensor(target).to(torch.float32)

            loss = F.cross_entropy(y_hat,target)
            loss.backward()
            optimizer.step() 
            optimizer.zero_grad()
            print(f"Done with batch {batchCount} on epoch {epochs} with a loss of {loss}")
            batchCount += 1
            if batchCount > testingRuns:
                return
        print(f"Done with epoch {epochs}")
        epochs -= 1
        
def main():
   
    if len(sys.argv) != 3:
        print("To properly run this .py you must argue two things.\nFirst, a path to the text file you would like to use for finetuning.\nSecond, either y or n for wheter or not you would like to load in an already tokenized version of the fine-tuning data.")
        exit()

    #load in data,tokenizer,and model
    corpus = load(sys.argv[1])
    tokenizer = AutoTokenizer.from_pretrained("distilgpt2")
    model = AutoModelForCausalLM.from_pretrained("distilgpt2")

    #tokenize the corpus
    if (sys.argv[2] == "n"):
        tokenizedCorpus = tokenize(corpus,tokenizer)
        np.save("savepoint",tokenizedCorpus)
    elif (sys.argv[2] == "y"):
        tokenizedCorpus = np.load("savepoint.npy")
    else:
        print("Bad parameter must be either y or n")

    #seperate into x and y
    x,y = seperate(tokenizedCorpus,128)

    #make dataset and dataloader
    dataset = TensorDataset(x,y)
    loader = DataLoader(dataset,batch_size=32,shuffle=True)

    tune(loader,model,1,0.0001,len(tokenizer),32,testing=True)
    model.save_pretrained("./model")
    print("Done With Training! Model is saved to ./model")

if __name__ == "__main__":
    main()    
