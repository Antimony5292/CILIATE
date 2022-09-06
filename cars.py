import pickle
import numpy as np
import os
import random
from torchvision import datasets, transforms

# Cifar100 data loader
class Cars:
    def __init__(self,inc_num):
        transform = transforms.Compose([transforms.Resize([224,224]),
#                               transforms.CenterCrop(224),
                              transforms.ToTensor()]
                                )
        self.exempler_num = 200
        self.inc_num = inc_num
        self.current_step = 0
        self.batch_num = inc_num
        self.train_data = datasets.ImageFolder('./data/Cars/train',transform=transform)
        self.test_data = datasets.ImageFolder('./data/Cars/test',transform=transform)
        self.base_cls = 196//self.inc_num
        self.train_groups, self.test_groups, self.val_groups = self.initialize()
#         self.train_groups = self.initialize()

        

    # Split into five groups for incremental learning
    def initialize(self):
        # split train data into 5 groups
        train_groups = [[] for g in range(self.inc_num+1)]
        for train_data, train_label in self.train_data:
#             train_data = train_data#.permute(1,2,0)
            # Split into five groups
            train_groups[train_label//self.base_cls].append((train_data,train_label))
            train_groups[self.inc_num-1].extend(train_groups[self.inc_num])


        # split test data into 5 groups
        test_groups = [[] for g in range(self.inc_num+1)]
        for test_data, test_label in self.test_data:
#             test_data = test_data#.permute(1,2,0)
            test_groups[test_label//self.base_cls].append((test_data,test_label))
            test_groups[self.inc_num-1].extend(test_groups[self.inc_num])


        # Build validation set with test images of old_classes and new_classes to 
        val_groups = [[],[],[],[],[]]
        for step in range(5):
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
        return self.train_groups[step_b], self.val_groups[step_b], self.test_groups[step_b]

if __name__ == "__main__":
    cifar = Cars(5)
    print(len(cifar.val_groups[cifar.inc_num-1]))
#     print(len(cifar.getNextClasses(0)[1]))
