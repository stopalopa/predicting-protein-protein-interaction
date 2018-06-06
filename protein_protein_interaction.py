import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
from torch import optim
import random
from torch.utils.data import DataLoader,Dataset
import torchvision.transforms as transforms
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
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
loss_history = []

class Config():
  train_epochs = 20
  learning_rate = .001
  batch_size = 4
  conv_filter_size = 12
  hidden_size = 1
  lstm_num_layers = 1
  fc_num_classes = 2
  margin = 2
  checkpoint = "2018"  #datetime.datetime.fromtimestamp(time.time()).strftime('%H:%M:%S')
  print_freq = 500
  checkpoint_dir = datetime.datetime.fromtimestamp(time.time()).strftime('%Y-%m-%d')
  kernel_size = 5
  pool_size = 2
  start_epoch = 0
  best_acc = 0
  drop_out = .2
  threshold = .5
  train_size = 0
  val_size = 0
  test_size = 0



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


def save_checkpoint(state, is_best):
  print(" ")
  filename="./" + Config.checkpoint_dir + "/" +Config.checkpoint + ".tar"
  if is_best:
      filename ="./" + Config.checkpoint_dir + "/best.tar"
  torch.save(state, filename)


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
  label = int(float(pair_label[1]))
  protein_seq_1 = proteins[protein_name_1]
  protein_seq_2 = proteins[protein_name_2]

  protein_1_tensor = Variable(aminoAcidstoTensor(protein_seq_1))
  # get seq_length x 1 x  21
  protein_2_tensor = Variable(aminoAcidstoTensor(protein_seq_2))
  protein_1_tensor = protein_1_tensor.unsqueeze(0)  # add a dimension for c_in
  protein_2_tensor = protein_2_tensor.unsqueeze(0)

  label_tensor = Variable(torch.LongTensor([categories.index(int(label))]).cuda())
  return protein_1_tensor, protein_2_tensor, label_tensor


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


def getPred(output1, output2):
  if F.pairwise_distance(output1.data, output2.data).cpu().numpy()[0] < Config.threshold:
    return 0
  else:
    return 1


class AverageMeter(object):
  def __init__(self):
    self.val = 0
    self.avg = 0
    self.tot_sum = 0
    self.count = 0
    
  def update(self, val):
    self.val = val
    self.tot_sum += val
    self.count += 1
    self.avg = self.tot_sum/self.count

def categoryFromOutput(output):
    top_n, top_i = output.topk(1)
    category_i = top_i.data.cpu().numpy()
    category_i = category_i[0][0]
    return categories[category_i], category_i
    

class MetricMeter(object):
  def __init__(self):
    self.count = 0
    self.tp = 0
    self.fp = 0
    self.tn = 0
    self.fn = 0
    self.y_true = []
    self.y_pred = []
    self.count = 0

  def update(self, guess, label):
    self.count += 1
    label_v = label.data
    label_val = label_v[0]
    if guess == 1 and label_val == 1:
      self.tp += 1
    elif guess == 1 and label_val == 0:
      self.fp += 1
    elif guess == 0 and label_val == 0:
      self.tn += 1
    elif guess == 0 and label_val == 1:
      self.fn += 1
    self.y_true.append(label_val)
    self.y_pred.append(guess)

      
def accuracy(guess, label):
  label_v = label.data
  label_val = label_v[0]

  if guess == label_val:
    return 1
  else:
    return 0


class Net(nn.Module):
  def __init__(self):
    super(Net, self).__init__()
    self.hidden_size = Config.hidden_size
    self.lstm_num_layers = Config.lstm_num_layers

    # Conv2d(in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=True)
    self.conv = nn.Conv2d(1, Config.conv_filter_size, (21, Config.kernel_size))
    self.conv_bn = nn.BatchNorm2d(Config.conv_filter_size)
    self.drop = nn.Dropout(p=Config.drop_out)
    # LSTM(self.hidden_size, self.hidden_size, batch_first=True, bidirectional=True)
    self.lstm = nn.LSTM(Config.conv_filter_size, self.hidden_size, self.lstm_num_layers, bidirectional=True)

    # Linear(in_features, out_features (num classes))
    self.fc = nn.Linear(self.hidden_size * 2, Config.fc_num_classes)

  def forward_once(self, x):
    # input: batch size x Cin x Height x Width  i.e. [1, 1, 21, 325]

    h0 = Variable(torch.zeros(self.lstm_num_layers * 2, 1, self.hidden_size))  # 2 for bidirection
    c0 = Variable(torch.zeros(self.lstm_num_layers * 2, 1, self.hidden_size))

    seq_len = x.size()[3]
    # self.conv.kernel_size = (seq_len, 21)

    x = self.conv(x)
    x = self.conv_bn(x)
    # x = F.max_pool2d(F.relu(x), (1, int(Config.conv_filter_size/2)))    #output: 1xoutchannelx1xsize_due_to_conv
    x = F.max_pool2d(F.relu(x), (1, Config.pool_size))
    #x = nn.MaxPool2d(F.relu(x), (1, Config.pool_size))
    x = self.drop(x)
    x = torch.squeeze(x, 2)  # output: 1xoutchannelxsize_due_to_convolution
    x = x.permute(2, 0, 1)  # output: conv_size x 1 x out_channels

    # input expected: (seq_len, batch, input_size)  batch first: (batch, seq, feature)
    out, (h_n, c_n) = self.lstm(x, (h0, c0))  # output (seq_len, batch, hidden_size * num_directions)
    # output (seq_len (conv_size), batch, hidden_size * num_directions) ie 54, 1, 2
    out = self.fc(out[-1, :, :])

    return out

  def forward(self, x1, x2):
        output1 = self.forward_once(x1)
        output2 = self.forward_once(x2)
        return output1, output2



def eval(net, criterion, optimizer, split):
  data_set = training_data
  if split == 'val':
    data_set = val_data
  elif split == 'test':
    data_set = test_data
  size_split = len(data_set)

  net.eval()
  losses = AverageMeter()
  accuracy_avg = AverageMeter()
  metrics = MetricMeter()
 
  for i in range(0, size_split):
    protein1, protein2, label = getProteinPair(i, split)
    output1, output2 = net(protein1, protein2)
    loss = criterion(output1, output2, label)
    guess = getPred(output1, output1)
    accuracy_avg.update(accuracy(guess, label))
    losses.update(loss.data[0])
    metrics.update(guess, label)
  return losses, metrics


def print_eval(losses, metrics, split, epoch_num):
  print(split, "losses", losses.avg)
  if metrics.tp + metrics.fn != 0:
    print(split,"sensitivity", (metrics.tp)/(metrics.tp + metrics.fn))
  if metrics.tn + metrics.fp != 0:
    print(split, "specificity", (metrics.tn)/(metrics.tn + metrics.fp))
  if metrics.tp + metrics.fp + metrics.fn + metrics.tn != 0:
    print(split, "accuracy", (metrics.tp + metrics.tn)/(metrics.tp + metrics.fp + metrics.fn + metrics.tn))
  if metrics.tp + metrics.fp != 0:
    print(split, "precision", (metrics.tp)/(metrics.tp + metrics.fp))

  cm = confusion_matrix(metrics.y_true, metrics.y_pred)
  print(cm)
  title = str(epoch_num + ' ' + split + ' Confusion Matrix')
  plt.ylabel('True Label')
  plt.xlabel('Predicated Label')
  plt.matshow(cm)
  fig_name = epoch_num + "_" + split + "_confusion_mat.png"
  plt.savefig(fig_name)

def train(net, criterion, optimizer, epoch):
  losses = AverageMeter()
  acc = AverageMeter()
  metrics = MetricMeter()
  Config.best_acc = 0

  size_train = len(training_data)
  size_train = 5
  if Config.train_size != 0:
    size_train = Config.train_size
  permutation = torch.randperm(len(training_data))
  for b in range(0, size_train, Config.batch_size):
    indices = permutation[b:b + Config.batch_size]
    total_loss = 0
    for p in indices:
      protein1, protein2, label = getProteinPair(p, "train")
      output1, output2 = net(protein1, protein2)
      contrastive_loss = criterion(output1, output2, label)
      total_loss = total_loss + contrastive_loss
      guess = getPred(output1, output1)
      acc.update(accuracy(guess, label))
      metrics.update(guess, label)
    avg_loss = total_loss/Config.batch_size
    avg_loss.backward()
    optimizer.step()
    optimizer.zero_grad()

    loss_history.append(contrastive_loss.data[0])
    is_best = acc.avg > Config.best_acc
    Config.best_acc = max(acc.avg, Config.best_acc)
    save_checkpoint({
      'epoch':epoch+1,
      'state_dict': net.state_dict(),
      'best_acc': Config.best_acc,
      'optimizer': optimizer.state_dict()}, is_best)

  return losses, acc, metrics



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
  parser.add_argument('--print_freq', default=10000)
  parser.add_argument('--resume', default=0)
  parser.add_argument('--desc', default="")
  parser.add_argument('--kernel_size', default=5)
  parser.add_argument('--pool_size', default=2)
  parser.add_argument('--drop_out', default=.2)
  parser.add_argument('--threshold', default=.5)
  parser.add_argument('--train_size', default=0)
  parser.add_argument('--val_size', default=0)
  parser.add_argument('--test_size', default=0)

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
  Config.kernel_size = int(args.kernel_size)
  Config.pool_size = int(args.pool_size)
  Config.start_epoch = 0
  Config.drop_out = float(args.drop_out)
  Config.threshold = float(args.threshold)
  Config.train_size = int(args.train_size)
  Config.val_size = int(args.val_size)
  Config.test_size = int(args.test_size)


  Config.checkpoint = "last"

  if not os.path.isdir(args.desc):
    os.makedirs(args.desc)

  Config.checkpoint_dir = args.desc
  torch.set_default_tensor_type('torch.cuda.FloatTensor')
           
  importData()

  net = Net().cuda()

  criterion = ContrastiveLoss().cuda()
  optimizer = optim.Adam(net.parameters(), lr=Config.learning_rate)

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
  
  trn_losses = None
  trn_metrics = None
  val_losses = None
  val_metrics = None
  test_losses = None
  test_metrics = None
                   
  for epoch in range(Config.start_epoch, Config.train_epochs):
    print("Epoch", epoch)
    trn_losses, trn_acc, trn_metrics = train(net, criterion, optimizer, epoch)
    for i in np.arange(0, 1, 0.1):
      print("train margin", i)
      train_losses, train_metrics = eval(net, criterion, optimizer, 'train')
      print_eval(train_losses, train_metrics, 'train', i)
    print("training acc", trn_acc.avg)
    if epoch % 3 == 0:
      for i in np.arange(0, 1, .1):
        print("val margin", i)
        Config.margin = i
        val_losses, val_metrics = eval(net, criterion, optimizer, 'val')
        print_eval(val_losses, val_metrics, 'val', i)  
    
  
  #trn_losses, trn_metrics = eval(net, criterion, optimizer, 'train')
  #print_eval(trn_losses, trn_metrics, 'train')
 
 
  #val_losses, val_metrics = eval_validate(net, criterion, optimizer, 'val')
  #print_eval(val_losses, val_metrics, 'val')


  #test_losses, test_metrics = eval_test(net, criterion, optimizer, 'test')
  #print_eval(test_losses, test_metrics, 'test')

if __name__ == "__main__":
    main()
