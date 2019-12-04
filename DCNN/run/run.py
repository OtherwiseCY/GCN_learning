from data import *
from model import *
from utils import *
from param import param

import numpy as np

def run():
    # load data
    adj,feat,label = data_parser()

    num_nodes = len(adj)
    indices = np.arange(num_nodes).astype('int32')
    np.random.shuffle(indices)

    ratio = param['train_val_test_ratio']
    total_ratio = np.sum(ratio)

    train_indices = indices[:ratio[0](*num_nodes)//total_ratio]
    valid_indices = indices[(ratio[0]*num_nodes//total_ratio):((ratio[0]+ratio[1])*num_nodes)//total_ratio]
    test_indices = indices[((ratio[0]+ratio[1])*num_nodes)//total_ratio:]

    # cal diffusion matrix
    diffA = compute_diffusion(adj,param['n_hop'])

    # data:np->tensor->dataset->dataloader
    trainA,trainY,valA,valY,testA,testY,X = map(torch.Tensor, 
                                                (diffA[train_indices],label[train_indices],
                                                diffA[valid_indices],label[valid_indices],
                                                diffA[test_indices],label[test_indices], feat))

    train_ds = data.TensorDataset(trainA,trainY)
    train_dl = data.DataLoader(train_ds, batch_size=32)

    # init model
    nodeClassDCNN = NodeClassDCNN(param)
    loss_func = param['loss_fn']
    opt = optim.SGD(nodeClassDCNN.parameters(),lr=param['lr'], momentum=param['momentum'])

    trainloss = []
    valloss = []
    trainacc = []
    valacc = []
    earlystop = 0

    # train
    for epoch in range(param['epoch']):
        nodeClassDCNN.train()
        for xb,yb in train_dl:
            pred = nodeClassDCNN(X,xb)
            loss = loss_func(pred,yb.long().squeeze())
            loss.backward()
            opt.step()
            opt.zero_grad()

        # eval
        nodeClassDCNN.eval()
        epochloss = loss_func(nodeClassDCNN(X,trainA), trainY.long().squeeze())
        accracy = np.mean((torch.argmax(nodeClassDCNN(X,trainA),1)==trainY.long().squeeze()).numpy())
        print(epoch,"epoch acc: {}\tloss: {}".format(accracy, epochloss))
        trainloss.append(epochloss)
        trainacc.append(accracy)
        
        epochloss = loss_func(nodeClassDCNN(X,valA), valY.long().squeeze())
        accracy = np.mean((torch.argmax(nodeClassDCNN(X,valA),1)==valY.long().squeeze()).numpy())
        valloss.append(epochloss)
        valacc.append(accracy)
        
        if len(valloss) > 2:
            if abs(valloss[-1]-valloss[-2]) <= param['earlystopping']:
                earlystop += 1
                if earlystop >= param['earlystopround']:
                    break
            else:
                earlystop = 0

    # save model
    torch.save(nodeClassDCNN.state_dict(),'../model/DCNN.pth')