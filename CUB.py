import pickle
import numpy as np
import os
import random
from torchvision import datasets, transforms

# Cifar100 data loader
class CUBS:
    def __init__(self):
        transform = transforms.Compose([transforms.Resize(255),
                              transforms.CenterCrop(224),
                              transforms.ToTensor()]
                                )
        self.exempler_num = 200
        self.current_step = 0
        self.batch_num = 5
        self.total_cls = 200
        self.train_data = datasets.ImageFolder('/data/gxq/Finetuning/data/CUBS/train',transform=transform)
        self.test_data = datasets.ImageFolder('/data/gxq/Finetuning/data/CUBS/test',transform=transform)
        
        self.train_groups, self.test_groups, self.val_groups = self.initialize()
#         self.train_groups = self.initialize()

        

    # Split into five groups for incremental learning
    def initialize(self):
        # split train data into 5 groups
        train_groups = [[],[],[],[],[]]
        base_cls = self.total_cls // self.batch_num
        for train_data, train_label in self.train_data:
            train_data = train_data#.permute(1,2,0)
            # Split into five groups
            if train_label < 40:
                train_groups[0].append((train_data,train_label))
            elif 40 <= train_label < 80:
                train_groups[1].append((train_data,train_label))
            elif 80 <= train_label < 120:
                train_groups[2].append((train_data,train_label))
            elif 120 <= train_label < 160:
                train_groups[3].append((train_data,train_label))
            elif 160 <= train_label < 200:
                train_groups[4].append((train_data,train_label))

        # split test data into 5 groups
        test_groups = [[],[],[],[],[]]
        for test_data, test_label in self.test_data:
            test_data = test_data#.permute(1,2,0)
            if test_label < 40:
                test_groups[0].append((test_data,test_label))
            elif 40 <= test_label < 80:
                test_groups[1].append((test_data,test_label))
            elif 80 <= test_label < 120:
                test_groups[2].append((test_data,test_label))
            elif 120 <= test_label < 160:
                test_groups[3].append((test_data,test_label))
            elif 160 <= test_label < 200:
                test_groups[4].append((test_data,test_label))

        # Build validation set with test images of old_classes and new_classes to 
        val_groups = [[],[],[],[],[]]
        for step in range(self.batch_num):
            # Calculate the number of new classes image and old classes image
            old_clases_propotion = step / (step + 1)
            new_clases_propotion = 1 / (step + 1)
            num_of_old_classes = int(self.exempler_num * old_clases_propotion)
            num_of_new_classes = int(self.exempler_num * new_clases_propotion)

            # Randomly fetch the old classes images
            if step >= 1:
                old_classes = []
                for i in range(step):
                    old_classes.extend(test_groups[i])
#                 assert(len(old_classes) == self.exempler_num * step)
                random.shuffle(old_classes)
                val_groups[step].extend(old_classes[:num_of_old_classes])
            
            # Randomly select new classes images
            random.shuffle(test_groups[step])
            val_groups[step].extend(test_groups[step][:num_of_new_classes])
#             assert(len(val_groups[step]) == self.exempler_num or len(val_groups[step]) == 1999)
    
        return train_groups, test_groups, val_groups

    # Return the data used for step_b
    def getNextClasses(self, step_b):
        return self.train_groups[step_b], self.val_groups[step_b]

if __name__ == "__main__":
    cifar = CUBS()
    for step_b in range(5):
        print(f"Incremental step : {step_b + 1}")

        # Get the train and val data for step b,
        # and split them into train_x, train_y, val_x, val_y
        train, val = cifar.getNextClasses(step_b)
        print(f'number of trainset: {len(train)}, number of valset: {len(val)}')
#     print(len(cifar.getNextClasses(0)[1]))
