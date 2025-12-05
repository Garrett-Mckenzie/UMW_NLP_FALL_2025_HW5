import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, TensorDataset, DataLoader
from transformers import AutoTokenizer, AutoModelForCausalLM
import sys
from tqdm import tqdm
from torch.optim import AdamW
import re

def load(filepath):
    corpus = None
    with open(filepath , 'r') as file:
        corpus = file.read()
        corpus = re.sub(r"<EOD>","",corpus)
        corpus = re.sub(r"</s>","",corpus)
        corpus = corpus.split("\n")
    for i in tqdm(range(len(corpus))):
        corpus[i] = corpus[i].strip() #removes whitespace
    return corpus

def tokenize(corpus,tokenizer):
    returnMe = np.array([])
    count = 1
    for i in tqdm(range(len(corpus))):
        doc = corpus[i]
        returnMe = np.append(returnMe,tokenizer(doc,return_tensors="np")["input_ids"].squeeze(0))
        count += 1
    return returnMe.astype(int)

def seperate(tokenizedCorpus,blockSize):
    length = len(tokenizedCorpus)
    train = []
    target = []
    pos = 0
    with tqdm(total = (len(tokenizedCorpus) // blockSize) - 1) as pbar:
        while True:
            if pos + blockSize + 1 >= length:
                break
            else:
                x = tokenizedCorpus[pos:pos + blockSize]
                y = tokenizedCorpus[pos + 1:pos + blockSize + 1]
                train.append(x)
                target.append(y)
                pos += blockSize
            pbar.update(1)
        
    if torch.cuda.is_available():
        x = torch.tensor(np.array(train)).to(torch.device("cuda:0"))
        y = torch.tensor(np.array(target))
        print("Training on GPU")
        return (x,y,True)
    else:
        print("No GPU available")
        return (torch.tensor(np.array(train)),torch.tensor(np.array(target)) , False)

def tune(loader,model,epochs,lr,vocabLength,batchSize,onGpu,testing=False,testingRuns=2):
    print("Starting Training")

    optimizer = AdamW(model.parameters(),lr = lr)
    count = 0

    while epochs > 0:
        batchCount = 0
        for x,y in loader:
            
            y_hat = model(input_ids=x).logits
            #This is kinda gross but it works
            target = np.zeros((x.shape[0] , len(y[0]) , vocabLength))

            for j in range(target.shape[0]):
                for i in range(target.shape[1]):
                    target[j][i][y[j][i]] += 1
            target = torch.tensor(target).to(torch.float32)

            if onGpu:
                target = target.to(torch.device("cuda:0"))

            loss = F.cross_entropy(y_hat,target)
            loss.backward()
            optimizer.step() 
            optimizer.zero_grad()

            #Move target back to cpu to clear up gpu
            target.detach().cpu()
            print(f"Done with batch {batchCount} on epoch {count} with a loss of {loss}")
            batchCount += 1
            if testing and batchCount > testingRuns:
                return
        print(f"Done with epoch {count}")
        torch.cuda.empty_cache()
        epochs -= 1
        count += 1

def main():

    #load in data,tokenizer,and model
    tokenizer = AutoTokenizer.from_pretrained("distilgpt2")
    model = AutoModelForCausalLM.from_pretrained("distilgpt2")

    #tokenize the corpus
    tokenizedCorpus = None
    if "--useSaved" in sys.argv:
        try:
            index = sys.argv.index("--useSaved")
            path = sys.argv[index + 1]
            tokenizedCorpus = np.load(path)    
        except Exception as e:
            print(e)
            print("if --useSaved is argued, a file path with the .npy extension must follow")
            return
    else:
        try:
            corpus = load(sys.argv[1])
            tokenizedCorpus = tokenize(corpus,tokenizer)
            np.save("savepoint",tokenizedCorpus)
        except Exception as e:
            print(e)
            print("if not providing a savepoint.npy a filepath with a .txt extension must be the only argument")
            return

    #seperate into x and y
    x,y,onGpu = seperate(tokenizedCorpus,128)

    if onGpu:
        model.to(torch.device("cuda:0"))

    #make dataset and dataloader
    dataset = TensorDataset(x,y)

    epochs = 5
    lr = 0.0001
    vocabLength = len(tokenizer)
    batchSize = 48
    testing = False
    testingRuns = 2
    if "--testingArgs" in sys.argv:
        try:
            index = sys.argv.index("--testingArgs")
            arguments = sys.argv[index + 1].split(":")
            epochs = int(arguments[0])
            lr = float(arguments[1])
            batchSize = int(arguments[2])
            testing = arguments[3]
            if testing == "1":
                testing = True
            else:
                testing = False
            testingRuns = int(arguments[4])
        except Exception as e:
            print(e)
            print("To specify --testingArgs follow the syntax --testingArgs epochs:lr:batchSize:testing:testingRuns")
            return

    loader = DataLoader(dataset,batch_size=batchSize,shuffle=True)
            
    #run the tuning job
    print("Using the below arguments for tuning")
    print("epochs: " + str(epochs))
    print("lr: " + str(lr))
    print("vocabLength: " + str(vocabLength))
    print("batchSize: " + str(batchSize))
    print("testing: " + str(testing))
    if testing:
        print("testingRuns: " + str(testingRuns))

    tune(loader,model,epochs,lr,vocabLength,batchSize,onGpu,testing=testing,testingRuns = testingRuns)
    model.save_pretrained("./model")
    print("Done With Training! Model is saved to ./model")

if __name__ == "__main__":
    main()    
