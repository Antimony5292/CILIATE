import torch
import torchvision
from torchvision.models import vgg16
from torchvision import transforms
from torch.utils.data import Dataset, TensorDataset, DataLoader, Subset, ConcatDataset
from torch.optim import Adam
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision.transforms import Compose, CenterCrop, Normalize, Resize, ToTensor, ToPILImage
from torchvision.models import resnet18
from torch.optim.lr_scheduler import MultiStepLR
# from torchsampler import ImbalancedDatasetSampler

import numpy as np
import PIL.Image as Image
import os
import gc
import pickle
import time
import random

from dataset import BatchData
from imbalanced import ImbalancedDatasetSampler
from model import Resnet, device
from cifar100 import Cifar100
from flowers import Flowers
from cars import Cars
from copy import deepcopy
from tqdm.notebook import tqdm


class Trainer:
    def __init__(self, total_cls, dataname, inc_num):
        self.total_cls = total_cls
        self.seen_cls = 0
        self.inc_num = inc_num
        self.dataname = dataname
        if dataname == 'flowers':
            self.dataset = Flowers()
            self.is_ori = False
            self.input_transform = Compose([
#                                     transforms.RandomHorizontalFlip(),
#                                     transforms.RandomCrop(224,padding=4),
#                                     ToTensor(),
#                                     Normalize([0.485, 0.456, 0.406],[0.229, 0.224, 0.225])
            ])

            self.input_transform_eval = Compose([
#                                     ToTensor(),
#                                     Normalize([0.485, 0.456, 0.406],[0.229, 0.224, 0.225])
            ])
            self.model = resnet18(pretrained=True)
            self.model.fc = nn.Linear(in_features=512, out_features=100, bias=True)
            self.model = self.model.to(device)
        elif dataname == 'CUBS':
            self.dataset = CUBS()
            self.is_ori = False
            self.input_transform = Compose([
                                    transforms.RandomHorizontalFlip(),
#                                     transforms.RandomCrop(224,padding=4),
#                                     ToTensor()
            ])

            self.input_transform_eval = Compose([
#                                     ToTensor()
            ])
            self.model = resnet18(pretrained=True)
            self.model.fc = nn.Linear(in_features=512, out_features=200, bias=True)
            self.model = self.model.to(device)
        elif dataname == 'cars':
            self.dataset = Cars(inc_num)
            self.is_ori = False
            self.input_transform = Compose([
                                    transforms.RandomHorizontalFlip(),
#                                     transforms.RandomCrop(224,padding=4),
#                                     ToTensor()
            ])

            self.input_transform_eval = Compose([
#                                     ToTensor()
            ])
            self.model = resnet18(pretrained=True)
            self.model.fc = nn.Linear(in_features=512, out_features=196, bias=True)
            self.model = self.model.to(device)           
        elif dataname == 'cifar100':
            self.is_ori = True
            self.dataset = Cifar100(inc_num)
            self.input_transform = Compose([
                                    transforms.RandomHorizontalFlip(),
                                    transforms.RandomCrop(32,padding=4),
                                    ToTensor(),
                                    Normalize([0.5071,0.4866,0.4409],[0.2673,0.2564,0.2762])])

            self.input_transform_eval = Compose([
                                    ToTensor(),
                                    Normalize([0.5071,0.4866,0.4409],[0.2673,0.2564,0.2762])])        
            self.model = Resnet(32,total_cls).to(device)
        print(self.model)


        total_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        print("Solver total trainable parameters : ", total_params)
        print("---------------------------------------------")
    def save_model(self,file_name):
        state = {
            'net': self.model.state_dict()
        }
        save_path = 'checkpoint'
        if not os.path.isdir(save_path):
            os.mkdir(save_path)
        torch.save(state, os.path.join(save_path,file_name))
        print('Model Saved!')
    
    def eval(self, valdata):
        self.model.eval()
        count = 0
        correct = 0
        wrong = 0
        for i, (image, label) in enumerate(valdata):
            image = image.to(device)
            label = label.view(-1).to(device)
            p = self.model(image)
#             print(p[:,:8])
            pred = p[:,:self.seen_cls].argmax(dim=-1)
#             print(pred)
            correct += sum(pred == label).item()
            wrong += sum(pred != label).item()
        acc = correct / (wrong + correct)
        print("Val Acc: {}".format(acc*100))
        self.model.train()
        print("---------------------------------------------")
        return acc

    # Get learning rate
    def get_lr(self, optimizer):
        for param_group in optimizer.param_groups:
            return param_group['lr']
    
    def train_1(self, model, train_data, batch_size, epoches, lr):
        criterion = nn.CrossEntropyLoss()

        # Set optimizer and scheduler
        optimizer = optim.SGD(model.parameters(), lr=lr, momentum=0.9,  weight_decay=2e-4)
        scheduler = MultiStepLR(optimizer, [100, 150, 200], gamma=0.1)

        # Print the number of classes have been trained


        for epoch in tqdm(range(epoches)):
            print("---------------------------------------------")
            print("Epoch", epoch)

            # Print current learning rate
            
            cur_lr = self.get_lr(optimizer)
            print("Current Learning Rate : ", cur_lr)

            # Train the model with KD
            model.train()
            self.stage1(model,train_data, criterion, optimizer, self.seen_cls) 
            scheduler.step()

    def upperbound_train(self, net_A, before_train_loader, before_test_loader, seen_cls=None, epoches=100):
        if seen_cls == None:
            seen_cls = self.total_cls
        def train_up(net,epoch,train_loader,map_list=None):
            print('\n[ Train epoch: %d ]' % epoch)
            net.train()
            train_loss = 0
            correct = 0
            total = 0
            for param in net.fc.parameters():
                grad = param
            gradient = torch.zeros(len(grad)).to(device)
            cnt = 0
            for batch_idx, (inputs, targets) in enumerate(train_loader):
        #         print(targets)
                if map_list is not None:
                    targets = torch.tensor([map_list[i.item()] for i in targets])
        #             print(targets)
                inputs, targets = inputs.to(device).float(), targets.to(device).long().view(-1)

                optimizer.zero_grad()

                benign_outputs = net(inputs)
                
                loss = criterion(benign_outputs[:,:seen_cls], targets)
        #         try:
        #             loss = criterion(benign_outputs, targets)
        #         except:
        #             print(batch_idx,targets)
                loss.backward()

                optimizer.step()
                train_loss += loss.item()
                _, predicted = benign_outputs.max(1)

                total += targets.size(0)
                correct += predicted.eq(targets).sum().item()

                for param in net.fc.parameters():
                    grad = param.grad
        #         print(grad.mean())
#                 gradient += grad
                cnt+=1

                if batch_idx % 10 == 0:
                    print('\nCurrent batch:', str(batch_idx))
                    print('Current benign train accuracy:', str(predicted.eq(targets).sum().item() / targets.size(0)))
                    print('Current benign train loss:', loss.item())

            print('\nTotal benign train accuarcy:', 100. * correct / total)
            print('Total benign train loss:', train_loss)
            return train_loss, gradient/cnt

        def test_up(net,epoch,test_loader,map_list=None):
            print('\n[ Test epoch: %d ]' % epoch)
            net.eval()
            benign_loss = 0
            benign_correct = 0
            total = 0
            with torch.no_grad():
                for batch_idx, (inputs, targets) in enumerate(test_loader):
                    if map_list is not None:
                        targets = torch.tensor([map_list[i.item()] for i in targets])
                    inputs, targets = inputs.to(device).float(), targets.to(device).long().view(-1)
                    total += targets.size(0)

                    outputs = net(inputs)
#                     print(outputs.shape)
#                     print(targets.shape)
                    loss = criterion(outputs[:,:seen_cls], targets)
                    benign_loss += loss.item()

                    _, predicted = outputs.max(1)
                    benign_correct += predicted.eq(targets).sum().item()

                    if batch_idx % 10 == 0:
                        print('\nCurrent batch:', str(batch_idx))
                        print('Current benign test accuracy:', str(predicted.eq(targets).sum().item() / targets.size(0)))
                        print('Current benign test loss:', loss.item())
            print('\nTotal benign test accuarcy:', 100. * benign_correct / total)
            print('Total benign test loss:', benign_loss)

            return benign_correct / total, benign_loss

        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(net_A.parameters(), lr=1e-4)
        for epoch in tqdm(range(0, epoches)):
            loss_hist, grad = train_up(net_A,epoch,before_train_loader)
            test_up(net_A,epoch,before_test_loader)
#             test_up(net_A,epoch,before_train_loader)
            
    def train_one_step(self, model, train_data, epoch=250, lr=0.1, opt='SGD'):
        def train(net,epoch,train_loader):
            print('\n[ Train epoch: %d ]' % epoch)
            print('Current learning rate: ',self.get_lr(optimizer))
            net.train()
            train_loss = 0
            correct = 0
            total = 0
            cnt = 0
            for batch_idx, (inputs, targets) in enumerate(train_loader):
                inputs, targets = inputs.to(device), targets.view(-1).to(device)
#                 print(targets)
                optimizer.zero_grad()

                benign_outputs = net(inputs)
                loss = criterion(benign_outputs, targets)
        #         try:
        #             loss = criterion(benign_outputs, targets)
        #         except:
        #             print(batch_idx,targets)
                loss.backward()

                optimizer.step()
                train_loss += loss.item()
                _, predicted = benign_outputs.max(1)

                total += targets.size(0)
                correct += predicted.eq(targets).sum().item()

                cnt+=1

                if batch_idx % 10 == 0:
                    print('\nCurrent batch:', str(batch_idx))
                    print('Current benign train accuracy:', str(predicted.eq(targets).sum().item() / targets.size(0)))
                    print('Current benign train loss:', loss.item())

            print('\nTotal benign train accuarcy:', 100. * correct / total)
            print('Total benign train loss:', train_loss)
            return train_loss

        criterion = nn.CrossEntropyLoss()
        if opt == 'SGD':
            optimizer = optim.SGD(model.parameters(), lr=lr, momentum=0.9,  weight_decay=2e-4)
        else:
            optimizer = optim.Adam(model.parameters(), lr=lr)
        scheduler = MultiStepLR(optimizer, [100, 150, 200], gamma=0.1)
#         train_data = DataLoader(train_dataset,batch_size=32, shuffle=True)
        for epoch in tqdm(range(0, epoch)):
            loss_hist = train(model,epoch,train_data)
            scheduler.step()
        return model
    
    def train(self, batch_size, epoches, lr, method, ita, loss_name, dropout_state, random_select):
        total_cls = self.total_cls
        criterion = nn.CrossEntropyLoss()
        begin_time = time.time()
        # Used for Knowledge Distill
        previous_model = None

        dataset = self.dataset
        val_xs = []
        val_ys = []
        train_xs = []
        train_ys = []

        test_accs = []
        
        if method == 'upperbound':
            before_train_loader = DataLoader(dataset.train_data,batch_size=batch_size,num_workers=8,shuffle=True,drop_last=True)
            before_test_loader = DataLoader(dataset.test_data,batch_size=batch_size,num_workers=8,shuffle=False,drop_last=True)
            self.upperbound_train(self.model,before_train_loader,before_test_loader, epoches=epoches)
            self.save_model('{}_{}{}'.format(self.dataname, method, begin_time))
            end_time = time.time()
            print(end_time-begin_time)            
            return 0
        
        for step_b in range(self.inc_num):
            print(f"Incremental step : {step_b + 1}")
            
            # Get the train and val data for step b,
            # and split them into train_x, train_y, val_x, val_y
#             train, val = dataset.getNextClasses(step_b)

            train, val, test = dataset.getNextClasses(step_b)
            print(f'number of trainset: {len(train)}, number of valset: {len(val)}')
            train_x, train_y = zip(*train)
            val_x, val_y = zip(*val)
            val_xs.extend(val_x)
            val_ys.extend(val_y)
            train_xs.extend(train_x)
            print('train_xs: ',len(train_xs))
            train_ys.extend(train_y)

            # Transform data and prepare dataloader
            train_data = DataLoader(BatchData(train_xs, train_ys, self.input_transform, is_ori=self.is_ori),
                        batch_size=batch_size, shuffle=True, drop_last=True)
#             train_data = DataLoader(BatchData(train_xs[:150], train_ys[:150], self.input_transform),
#                         batch_size=batch_size, shuffle=True, drop_last=True)            
            val_data = DataLoader(BatchData(val_xs, val_ys, self.input_transform_eval, is_ori=self.is_ori),
                        batch_size=batch_size, shuffle=False)
            
            # Set optimizer and scheduler
            optimizer = optim.SGD(self.model.parameters(), lr=lr, momentum=0.9,  weight_decay=2e-4)

#             optimizer = optim.Adam(self.model.parameters(), lr=lr)
#             scheduler = MultiStepLR(optimizer, [300], gamma=0.1)
            scheduler = MultiStepLR(optimizer, [100, 150, 200], gamma=0.1)
            # Print the number of classes have been trained
            self.seen_cls += total_cls//self.inc_num
            if step_b == self.inc_num-1:
                self.seen_cls = total_cls
            print("seen classes : ", self.seen_cls)
            test_acc = []
            self.model.train()
            if method == 'FN':
                if step_b == 0:
                    if self.dataname == 'cifar100':   
                        for epoch in tqdm(range(epoches)):
                            self.stage1(self.model, train_data, criterion, optimizer, self.seen_cls) 
                            scheduler.step()
                    else:
                        self.upperbound_train(self.model,train_data,val_data, self.seen_cls)
                else:
                    if random_select:
                        adv_data, benign_data = self.get_random_adv(batch_size,train_data,ita=ita)
                    else:
                        model_A = deepcopy(self.model)
                        model_B = deepcopy(self.model)
    #                     self.train_one_step(model_A,train_data,epoch=epoches)
                        self.upperbound_train(model_A,train_data,val_data, self.seen_cls)
                        adv_data, benign_data = self.get_loss_adv(model_A,model_B,batch_size,train_data,ita=ita,loss_name=loss_name)
                    if self.dataname == 'cifar100': 
                        optimizer = optim.SGD(self.model.parameters(), lr=lr, momentum=0.9,  weight_decay=2e-4)
                    else:
                        optimizer = optimizer = optim.Adam(self.model.parameters(), lr=1e-4)
                    self.model.train()
#                     optimizer = optim.Adam(model.parameters(), lr=lr,  weight_decay=2e-4)
                    scheduler = MultiStepLR(optimizer, [100, 150, 200], gamma=0.1)
                    for epoch in tqdm(range(epoches)):
                        scheduler.step()
                        cur_lr = self.get_lr(optimizer)
#                         print("Current Learning Rate : ", cur_lr)
                        self.model.dropout_on = True
                        if dropout_state == 'ordinary':
                            self.model.dropout_on = False
                        self.stage1_distill(self.model, adv_data, criterion, optimizer, self.seen_cls)
                    # benign training
                    if self.dataname == 'cifar100': 
                        optimizer = optim.SGD(self.model.parameters(), lr=lr, momentum=0.9,  weight_decay=2e-4)
                    else:
                        optimizer = optimizer = optim.Adam(self.model.parameters(), lr=1e-4)
                    scheduler = MultiStepLR(optimizer, [100, 150, 200], gamma=0.1)
                    for epoch in tqdm(range(epoches)):
                        scheduler.step()
                        cur_lr = self.get_lr(optimizer)
#                         print("Current Learning Rate : ", cur_lr)
                        self.model.dropout_on = False
                        if dropout_state == 'dropout':
                            self.model.dropout_on = True
                        self.stage1_distill(self.model, benign_data, criterion, optimizer, self.seen_cls)                    
            elif method == 'WA':
                for epoch in range(epoches):
                    print("---------------------------------------------")
                    print("Epoch", epoch)

                    # Print current learning rate
                    scheduler.step()
                    cur_lr = self.get_lr(optimizer)
                    print("Current Learning Rate : ", cur_lr)

                    # Train the model with KD
                    self.model.train()
                    if step_b >= 1:
                        self.stage1_distill(self.model,train_data, criterion, optimizer,self.seen_cls)
                    else:
                        self.stage1(self.model,train_data, criterion, optimizer,self.seen_cls)

                    # Evaluation
#                     acc = self.eval(val_data)
                    acc = self.eval(train_data)

                if step_b >= 1:
                    self.model.weight_align(step_b)
            elif method == 'icarl':
                for epoch in range(epoches):
                    print("---------------------------------------------")
                    print("Epoch", epoch)

                    # Print current learning rate
                    scheduler.step()
                    cur_lr = self.get_lr(optimizer)
                    print("Current Learning Rate : ", cur_lr)

                    # Train the model with KD
                    self.model.train()
                    if step_b >= 1:
                        self.stage1_distill(self.model,train_data, criterion, optimizer,self.seen_cls)
                    else:
                        self.stage1(self.model,train_data, criterion, optimizer,self.seen_cls)

                    # Evaluation
#                     acc = self.eval(val_data)
                    acc = self.eval(train_data)            
            elif method == 'upperbound':
                if step_b == self.inc_num-1:
                    self.upperbound_train(self.model,train_data,val_data,epoches=epoches)
                
            # deepcopy the previous model used for KD
            self.previous_model = deepcopy(self.model)

            # Evaluate final accuracy at the end of one batch
            acc = self.eval(val_data)
            test_accs.append(acc)
            
            print(f'Previous accuracies: {test_accs}')
            if method == 'FN':
                self.save_model('{}_{}FN{}_{}_{}'.format(self.dataname, ita, begin_time, self.inc_num, step_b))
            else:
                self.save_model('{}_{}{}_{}_{}'.format(self.dataname, method, begin_time, self.inc_num, step_b))
        end_time = time.time()
        print(end_time-begin_time)
            
    def stage1(self, model, train_data, criterion, optimizer, seen_cls):
#         print("Training ... ")
        losses = []
        for i, (image, label) in enumerate(train_data):
            image = image.to(device)
            label = label.view(-1).to(device)
            p = model(image)
            loss = criterion(p[:,:seen_cls], label)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            losses.append(loss.item())
        print("CE loss :", np.mean(losses))

    def stage1_distill(self, model, train_data, criterion, optimizer,seen_cls):
#         print("Training ... ")
        distill_losses = []
        ce_losses = []
        T = 2
        beta = (seen_cls - 20)/self.seen_cls
#         print("classification proportion 1-beta = ", 1-beta)
        for i, (image, label) in enumerate(train_data):
            image = image.to(device)
            label = label.view(-1).to(device)
            p = model(image)
            with torch.no_grad():
                previous_q = self.previous_model(image)
                previous_q = F.softmax(previous_q[:,:self.seen_cls-20]/T, dim=1)
            log_current_p = F.log_softmax(p[:,:seen_cls-20]/T, dim=1)
            loss_distillation = -torch.mean(torch.sum(previous_q * log_current_p, dim=1))
            loss_crossEntropy = nn.CrossEntropyLoss()(p[:,:seen_cls], label)
            loss = loss_distillation * T * T + (1-beta) * loss_crossEntropy
            optimizer.zero_grad()
            loss.backward(retain_graph=True)
            optimizer.step()
            distill_losses.append(loss_distillation.item())
            ce_losses.append(loss_crossEntropy.item())
        print("KD loss :", np.mean(distill_losses), "; CE loss :", np.mean(ce_losses))
    
    def get_loss_adv(self, net_A,net_B,batch_size,all_train_loader,ita=0.1,loss_name='JS'):
        FT_rate_list = []
        if loss_name == 'KL':
            loss = nn.KLDivLoss(reduction='none')
        elif loss_name == 'JS':
            def loss(p_output, q_output):
                KLDivLoss = nn.KLDivLoss(reduction='none')
                p_output = F.softmax(p_output)
                q_output = F.softmax(q_output)    
                log_mean_output = ((p_output + q_output )/2).log()
                JSD = (KLDivLoss(log_mean_output, p_output) + KLDivLoss(log_mean_output, q_output))/2
                return JSD

        for x,y in tqdm(all_train_loader):
            gc.collect()
            net_A,net_B = net_A.to(device),net_B.to(device)
            x = torch.tensor(x,dtype=torch.float32).to(device)
            output_A = net_A(x)
            output_B = net_B(x)
            FT_rate = loss(output_A[:,:self.seen_cls],output_B[:,:self.seen_cls])
            FT_rate = torch.sum(FT_rate, dim=1, keepdim=True)
    #             FT_relative_rate = FT_rate-FT_rate.mean()
            FT_rate_list.extend(list(FT_rate.clone().detach().cpu()))
        FT_rate_np = torch.cat(FT_rate_list).numpy()
        idx_list = np.argsort(-FT_rate_np) # descending sort
        adv_idx = idx_list[:int(ita*len(idx_list))]

        benign_idx = []
        cnt = 0
        for i in tqdm(range(len(all_train_loader.dataset))):
            if i not in adv_idx:
                benign_idx.append(i)
                
        adv_dataset = Subset(all_train_loader.dataset,adv_idx)
        adv_x, adv_y = zip(*adv_dataset)
        adv_xs = []
        adv_ys = []
        adv_xs.extend(adv_x)
        adv_ys.extend(adv_y)
#         adv_dataset = TensorDataset(torch.tensor(adv_xs),torch.tensor(adv_ys))
        adv_dataset = BatchData(adv_xs,adv_ys,is_ori=False)
        print(len(adv_dataset.labels))
        adv_l = []
        for y in adv_dataset.labels:
            adv_l.append(y[0].item())
        benign_dataset = Subset(all_train_loader.dataset,benign_idx)
        benign_x, benign_y = zip(*benign_dataset)
        benign_xs = []
        benign_ys = []
        benign_xs.extend(benign_x)
        benign_ys.extend(benign_y)
        benign_dataset = BatchData(benign_xs,benign_ys,is_ori=False)
#         benign_dataset = TensorDataset(torch.tensor(benign_xs),torch.tensor(benign_ys))
        benign_l = []
        for y in benign_dataset.labels:
            benign_l.append(y[0].item())
        print(len(benign_dataset.labels))
        adv_dataloader = DataLoader(adv_dataset,sampler=ImbalancedDatasetSampler(adv_dataset,adv_l),batch_size=batch_size)
        benign_dataloader = DataLoader(benign_dataset,sampler=ImbalancedDatasetSampler(benign_dataset,benign_l),batch_size=batch_size)

        return adv_dataloader,benign_dataloader
    def get_random_adv(self,batch_size,all_train_loader,ita=0.1):
        adv_idx = random.choices(range(0,len(all_train_loader.dataset)),k=int(ita*len(all_train_loader.dataset)))

        benign_idx = []
        cnt = 0
        for i in tqdm(range(len(all_train_loader.dataset))):
            if i not in adv_idx:
                benign_idx.append(i)

        adv_dataset = Subset(all_train_loader.dataset,adv_idx)
        adv_x, adv_y = zip(*adv_dataset)
        adv_xs = []
        adv_ys = []
        adv_xs.extend(adv_x)
        adv_ys.extend(adv_y)
    #         adv_dataset = TensorDataset(torch.tensor(adv_xs),torch.tensor(adv_ys))
        adv_dataset = BatchData(adv_xs,adv_ys,is_ori=False)
        print(len(adv_dataset.labels))
        adv_l = []
        for y in adv_dataset.labels:
            adv_l.append(y[0].item())
        benign_dataset = Subset(all_train_loader.dataset,benign_idx)
        benign_x, benign_y = zip(*benign_dataset)
        benign_xs = []
        benign_ys = []
        benign_xs.extend(benign_x)
        benign_ys.extend(benign_y)
        benign_dataset = BatchData(benign_xs,benign_ys,is_ori=False)
    #         benign_dataset = TensorDataset(torch.tensor(benign_xs),torch.tensor(benign_ys))
        benign_l = []
        for y in benign_dataset.labels:
            benign_l.append(y[0].item())
        print(len(benign_dataset.labels))
        adv_dataloader = DataLoader(adv_dataset,sampler=ImbalancedDatasetSampler(adv_dataset,adv_l),batch_size=batch_size)
        benign_dataloader = DataLoader(benign_dataset,sampler=ImbalancedDatasetSampler(benign_dataset,benign_l),batch_size=batch_size)

        return adv_dataloader,benign_dataloader    