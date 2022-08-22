import pickle
import numpy as np
import os
import random

# Cifar100 data loader
class Cifar100:
    def __init__(self):
        with open('cifar-100-python/train','rb') as f:
            self.train = pickle.load(f, encoding='latin1')
        with open('cifar-100-python/test','rb') as f:
            self.test = pickle.load(f, encoding='latin1')
        self.train_data = self.train['data']
        self.train_labels = self.train['fine_labels']
        self.test_data = self.test['data']
        self.test_labels = self.test['fine_labels']
        self.train_groups, self.test_groups, self.val_groups = self.initialize()
        self.current_step = 0
        self.batch_num = 5

    # Split into five groups for incremental learning
    def initialize(self):
        # split train data into 5 groups
        train_groups = [[],[],[],[],[]]
        for train_data, train_label in zip(self.train_data, self.train_labels):
            # Decode the data from binary form
            train_data_r = train_data[:1024].reshape(32, 32)
            train_data_g = train_data[1024:2048].reshape(32, 32)
            train_data_b = train_data[2048:].reshape(32, 32)
            train_data = np.dstack((train_data_r, train_data_g, train_data_b))

            # Split into five groups
            if train_label < 20:
                train_groups[0].append((train_data,train_label))
            elif 20 <= train_label < 40:
                train_groups[1].append((train_data,train_label))
            elif 40 <= train_label < 60:
                train_groups[2].append((train_data,train_label))
            elif 60 <= train_label < 80:
                train_groups[3].append((train_data,train_label))
            elif 80 <= train_label < 100:
                train_groups[4].append((train_data,train_label))

        # split test data into 5 groups
        test_groups = [[],[],[],[],[]]
        for test_data, test_label in zip(self.test_data, self.test_labels):
            test_data_r = test_data[:1024].reshape(32, 32)
            test_data_g = test_data[1024:2048].reshape(32, 32)
            test_data_b = test_data[2048:].reshape(32, 32)
            test_data = np.dstack((test_data_r, test_data_g, test_data_b))

            if test_label < 20:
                test_groups[0].append((test_data,test_label))
            elif 20 <= test_label < 40:
                test_groups[1].append((test_data,test_label))
            elif 40 <= test_label < 60:
                test_groups[2].append((test_data,test_label))
            elif 60 <= test_label < 80:
                test_groups[3].append((test_data,test_label))
            elif 80 <= test_label < 100:
                test_groups[4].append((test_data,test_label))

        # Build validation set with test images of old_classes and new_classes to 
        val_groups = [[],[],[],[],[]]
        for step in range(5):
            # Calculate the number of new classes image and old classes image
            old_clases_propotion = step / (step + 1)
            new_clases_propotion = 1 / (step + 1)
            num_of_old_classes = int(2000 * old_clases_propotion)
            num_of_new_classes = int(2000 * new_clases_propotion)

            # Randomly fetch the old classes images
            if step >= 1:
                old_classes = []
                for i in range(step):
                    old_classes.extend(test_groups[i])
                assert(len(old_classes) == 2000 * step)
                random.shuffle(old_classes)
                val_groups[step].extend(old_classes[:num_of_old_classes])
            
            # Randomly select new classes images
            random.shuffle(test_groups[step])
            val_groups[step].extend(test_groups[step][:num_of_new_classes])
            assert(len(val_groups[step]) == 2000 or len(val_groups[step]) == 1999)
    
        return train_groups, test_groups, val_groups

    # Return the data used for step_b
    def getNextClasses(self, step_b):
        return self.train_groups[step_b], self.val_groups[step_b]

if __name__ == "__main__":
    cifar = Cifar100()
    print(len(cifar.train_groups[0]))
    print(len(cifar.getNextClasses(0)[1]))
