import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
from torch import optim
import random
from torch.utils.data import DataLoader,Dataset
import torchvision.transforms as transforms
import numpy as np
import matplotlib.pyplot as plt
import argparse
import datetime
import shutil
import time
import os


char_to_index = {'M': 0, 'S': 1, 'V': 2, 'E': 3, 'D': 4, 'F': 5, 'I': 6, 'Q': 7, 'P': 8, 'Y': 9, 'T': 10, 'L': 11, 'N': 12, 'R': 13, 'G': 14, 'K': 15, 'C': 16, 'H': 17, 'A': 18, 'W': 19, 'U': 20}
index_to_char = {0: 'M', 1: 'S', 2: 'V', 3: 'E', 4: 'D', 5: 'F', 6: 'I', 7: 'Q', 8: 'P', 9: 'Y', 10: 'T', 11: 'L', 12: 'N', 13: 'R', 14: 'G', 15: 'K', 16: 'C', 17: 'H', 18: 'A', 19: 'W', 20: 'U'}
n_letters = len(char_to_index)

categories = [0, 1]
proteins = {}
training_data = []
val_data = []
test_data = []


class Config():
  train_epochs = 20
  learning_rate = .001
  batch_size = 1
  conv_filter_size = 12
  hidden_size = 1
  lstm_num_layers = 1
  fc_num_classes = 2
  margin = 2
  checkpoint = "2018"  #datetime.datetime.fromtimestamp(time.time()).strftime('%H:%M:%S')
  print_freq = 500
  checkpoint_dir = datetime.datetime.fromtimestamp(time.time()).strftime('%Y-%m-%d')

  
def letterToIndex(letter):
  return char_to_index[letter]

def letterToTensor(letter):
  tensor = torch.zeros(1, n_letters).cuda()
  tensor[0][letterToIndex(letter)] = 1
  return tensor

def aminoAcidstoTensor(aminoAcidSeq):
  tensor = torch.zeros(1, n_letters, len(aminoAcidSeq)).cuda()  #torch.zeros(len(aminoAcidSeq), 1, n_letters)
  for idx, letter in enumerate(aminoAcidSeq):
    tensor[0][letterToIndex(letter)][idx] = 1
  return tensor


def importData(): 
  with open("protein_amino_acid_mapping.txt") as protein_data:
    for line in protein_data:
      line = line.strip('\n')
      name_seq_pair = line.split(' ')
      proteins[name_seq_pair[0]] = name_seq_pair[1]
                                         
  protein_data.close()

  with open('train_file.txt') as train_data:
    for line in train_data:
      line = line.strip('\n')
      training_data.append(line)
  train_data.close()

  with open('val_file.txt') as validation_data:
    for line in validation_data:
      line = line.strip('\n')
      val_data.append(line)
  validation_data.close()

  with open('test_file.txt') as testing_data:
    for line in testing_data:
      line = line.strip('\n')
      test_data.append(line)

  testing_data.close()

def save_checkpoint(state, is_best, filename="./" + Config.checkpoint_dir + "/" +Config.checkpoint + ".tar"):
  if is_best:
      filename ="./" + Config.checkpoint_dir + "/best_" + Config.checkpoint + ".tar"
  torch.save(state, filename)

    
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.hidden_size = Config.hidden_size
        self.lstm_num_layers = Config.lstm_num_layers
        
        #Conv2d(in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=True)
        self.conv = nn.Conv2d(1, Config.conv_filter_size, (21, 1))
        
        #LSTM(self.hidden_size, self.hidden_size, batch_first=True, bidirectional=True)
        self.lstm = nn.LSTM(12, self.hidden_size, self.lstm_num_layers, bidirectional=True)
        
        # Linear(in_features, out_features (num classes))
        self.fc = nn.Linear(self.hidden_size * 2, Config.fc_num_classes)


    def forward_once(self, x):
        #input: batch size x Cin x Height x Width  i.e. [1, 1, 21, 325]
        
        h0 = Variable(torch.zeros(self.lstm_num_layers * 2, 1, self.hidden_size))  # 2 for bidirection
        c0 = Variable(torch.zeros(self.lstm_num_layers * 2, 1, self.hidden_size))
        
        seq_len = x.size()[3]
        self.conv.kernel_size = (seq_len, 21)
        x = self.conv(x)
        x = F.max_pool2d(F.relu(x), (1, 6))    #output: 1xoutchannelx1xsize_due_to_conv
        x = torch.squeeze(x, 2)                #output: 1xoutchannelxsize_due_to_convolution
        x = x.permute(2, 0, 1)                 #output: conv_size x 1 x out_channels
    
        #input expected: (seq_len, batch, input_size)  batch first: (batch, seq, feature)
        out, (h_n, c_n) = self.lstm(x, (h0, c0)) #output (seq_len, batch, hidden_size * num_directions)
        #output (seq_len (conv_size), batch, hidden_size * num_directions) ie 54, 1, 2
        out = self.fc(out[-1, :, :])
       
        return out


    def forward(self, x1, x2):
        output1 = self.forward_once(x1)
        output2 = self.forward_once(x2)
        return output1, output2

def getProteinPair(index, data_set):
  if data_set == 'train':
    sample = training_data[index]
  elif data_set == 'val':
    sample = val_data[index]
  else:
    sample = test_data[index]
  pair_label = sample.split(' ')
  protein_pair = pair_label[0].split('-')
  protein_name_1 = protein_pair[0]
  protein_name_2 = protein_pair[1]
  label = pair_label[1]
  
  protein_seq_1 = proteins[protein_name_1]
  protein_seq_2 = proteins[protein_name_2]
    
  #Variable(aminoAcidstoTensor(protein_seq_1)).cuda()
  
  protein_1_tensor = Variable(aminoAcidstoTensor(protein_seq_1))
  #get seq_length x 1 x  21
  protein_2_tensor = Variable(aminoAcidstoTensor(protein_seq_2))
  protein_1_tensor = protein_1_tensor.unsqueeze(0)  #add a dimension for c_in
  protein_2_tensor = protein_2_tensor.unsqueeze(0)

  label_tensor = Variable(torch.LongTensor([categories.index(int(label))]).cuda())
  return protein_1_tensor, protein_2_tensor, label_tensor

class AverageMeter(object):
  def __init__(self):
    self.val = 0
    self.avg = 0
    self.tot_sum = 0
    self.count = 0

  def reset(self):
    self.val = 0
    self.avg = 0
    self.tot_sum = 0
    self.count = 0
    
  def update(self, val):
    self.val = val
    self.tot_sum += val
    self.count += 1
    self.avg = self.tot_sum/self.count
  


class ContrastiveLoss(torch.nn.Module):
    """
    Contrastive loss function.
    Based on: http://yann.lecun.com/exdb/publis/pdf/hadsell-chopra-lecun-06.pdf
    """

    def __init__(self):
        super(ContrastiveLoss, self).__init__()

    def forward(self, output1, output2, label):
        euclidean_distance = F.pairwise_distance(output1, output2)
        label = label.type(torch.FloatTensor).cuda()
        loss_contrastive = torch.mean((1-label) * torch.pow(euclidean_distance, 2).cuda() +
                                      (label) * torch.pow(torch.clamp(Config.margin - euclidean_distance, min=0.0), 2)).cuda()
       
        return loss_contrastive
      

def accuracy(output1, output2, label):
  if ((F.pairwise_distance(output1.data, output2.data) < Config.margin).cpu().numpy() and label.data.cpu().numpy()==1):
    return 1
  else:
    return 0
  
def validate(net, criterion, optimizer):
  net.eval()
  losses = AverageMeter()
  accuracy_avg = AverageMeter()
  
  size_val = len(val_data)
  acc = 0
  size_val = 3
  for i in range(0, size_val):
    protein1, protein2, label = getProteinPair(i, "val")
    output1, output2 = net(protein1, protein2)
    loss = criterion(output1, output2, label)
    acc += accuracy(output1, output2, label)
    accuracy_avg.update(acc)
    losses.update(loss.data[0])

    if i % Config.print_freq == 0:
      print('Loss value', losses.val)
      print('Loss average', losses.avg) 

  print ("Validation accuracy", accuracy_avg.avg)         


def train(net, criterion, optimizer):
  losses = AverageMeter()
  acc = 0
 # f = open('{0}_{1}_{2}.csv'.format(Config.train_number_epochs, Config.learning_rate, Config.margin), 'w')
  best_acc = 0
  for epoch in range(Config.train_epochs): 
    print("epoch ", epoch)
    size_train = len(train_data)
    print("accuracy", acc/size_train)
    acc = 0
    for i in range(0, size_train):
      if i % Config.print_freq == 0:
        print(str(datetime.datetime.now())) #11:45 shows as 3:45
        print(i)
      counter.append(i)
      protein1, protein2, label = getProteinPair(i, "train")
      output1, output2 = net(protein1, protein2)
      #question: how does this work? _, preds = torch.max(outputs.data, 1)
      #output: size_conv x 2 

      optimizer.zero_grad()
      contrastive_loss = criterion(output1, output2, label)
      contrastive_loss.backward()
      optimizer.step()
      acc += accuracy(output1, output2, label)
      if i % Config.print_freq == 0:
        print(str(datetime.datetime.now()))
        print("sample", i)
      
    loss_history.append(contrastive_loss.data[0])
    print("loss", loss_history)
    is_best = acc/size_train > best_acc
    best_acc = max(acc/size_train, best_acc)
    save_checkpoint({
      'epoch':epoch+1,
      'state_dict': net.state_dict(),
      'best_acc': best_acc,
      'optimizer': optimizer.state_dict()}, is_best)
                    
 # loss_hist_str = ' '.join(str(loss_history) for l in loss)
 # f.write(loss_hist_str)
  #plt.plot(counter,loss_history)
  #plt.show()

counter = []
loss_history = []
iteration_number = 0

def main():
  
  parser = argparse.ArgumentParser()
  parser.add_argument('--train_epochs', default=20)
  parser.add_argument('--learning_rate', default=.001)
  parser.add_argument('--margin', default=2)
  parser.add_argument('--batch_size', default=1)
  parser.add_argument('--conv_filter_size', default=12)
  parser.add_argument('--hidden_size', default=1)
  parser.add_argument('--lstm_num_layers', default=1)
  parser.add_argument('--fc_num_classes', default=2)
  parser.add_argument('--print_freq', default=1000)
  parser.add_argument('--resume', default=0)
  
  args = parser.parse_args()
  Config.train_epochs = int(args.train_epochs)
  Config.learning_rate = float(args.learning_rate)
  Config.margin = float(args.margin)
  Config.batch_size = int(args.batch_size)
  Config.conv_filter_size = int(args.conv_filter_size)
  Config.hidden_size = int(args.hidden_size)
  Config.lstm_num_layers = int(args.lstm_num_layers)
  Config.fc_num_classes = int(args.fc_num_classes)
  Config.print_freq = int(args.print_freq)


  ts = datetime.datetime.fromtimestamp(time.time()).strftime('%Y-%m-%d_%H:%M:%S')

  
  Config.checkpoint = '{}_{}_{}_{}_{}_{}_{}_{}_{}'.format(ts, args.train_epochs, str(args.learning_rate)[2:], str(args.margin).replace('.', ''), args.batch_size, args.conv_filter_size, args.hidden_size, args.lstm_num_layers, args.fc_num_classes)
  print(Config.checkpoint)

  ts = datetime.datetime.fromtimestamp(time.time()).strftime('%Y-%m-%d')

  if not os.path.isdir(ts):
    os.makedirs(ts)

  Config.checkpoint_dir = ts
  torch.set_default_tensor_type('torch.cuda.FloatTensor')

  importData() 
  net = Net().cuda()
  net = net
  criterion = ContrastiveLoss().cuda()
  optimizer = optim.Adam(net.parameters(), Config.learning_rate)

   
  if args.resume:
    if os.path.isfile(args.resume):
      print("=> loading checkpoint '{}'".format(args.resume))
      checkpoint = torch.load(args.resume)
      args.start_epoch = checkpoint['epoch']
      best_acc = checkpoint['best_acc']
      net.load_state_dict(checkpoint['state_dict'])
      optimizer.load_state_dict(checkpoint['optimizer'])
      print("=> loaded checkpoint '{}' (epoch {})".format(args.resume, checkpoint['epoch']))
    else:
      print("=> no checkpoint found at '{}'".format(args.resume))
   

#get protein of dimensions: batch x height(21) x seq_length
  train(net, criterion, optimizer)
  validate(net, criterion, optimizer)

if __name__ == "__main__":
    main()
