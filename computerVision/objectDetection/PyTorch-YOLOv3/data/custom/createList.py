import os, random

data = os.listdir('images/')

random.shuffle(data)

numTrain = int(0.75*len(data))

train_data = data[:numTrain]
val_data = data[numTrain:]

with open('train.txt', 'w') as f1:
    for data in train_data:
        f1.write(os.path.join('data/custom/images', data) + '\n')
                            
with open('valid.txt', 'w') as f2:
    for data in val_data:
        f2.write(os.path.join('data/custom/images', data) + '\n')
