
# coding: utf-8

# In[ ]:


# Importing the required libraries

import numpy as np
import torch
from torch import nn
import torch.nn.functional as F

# Loading the data
# open text file and read in data as `text`
with open('data/Bhagavad-gita_As_It_Is.txt', 'r') as f:
    text = f.read()
    

# 1. int2char, which maps integers to characters
# 2. char2int, which maps characters to unique integers
chars = tuple(set(text))
int2char = dict(enumerate(chars))
char2int = {ch: ii for ii, ch in int2char.items()}

# encoding the text
encoded = np.array([char2int[ch] for ch in text])

def one_hot_encode(arr, n_labels):
    
    # Initialize the the encoded array
    one_hot = np.zeros((np.multiply(*arr.shape), n_labels), dtype=np.float32)
    
    # Fill the appropriate elements with ones
    one_hot[np.arange(one_hot.shape[0]), arr.flatten()] = 1.
    
    # Finally reshape it to get back to the original array
    one_hot = one_hot.reshape((*arr.shape, n_labels))
    
    return one_hot

# checking that the function works as expected
test_seq = np.array([[2, 3, 1]])
one_hot = one_hot_encode(test_seq, 8)

print(one_hot)
"""
[[[0. 0. 1. 0. 0. 0. 0. 0.]
  [0. 0. 0. 1. 0. 0. 0. 0.]
  [0. 1. 0. 0. 0. 0. 0. 0.]]]
"""
def get_batches(arr, batch_size, seq_length):
    '''Create a generator that returns batches of size
       batch_size x seq_length from arr.
       
       Arguments
       ---------
       arr: Array you want to make batches from
       batch_size: Batch size, the number of sequences per batch
       seq_length: Number of encoded chars in a sequence
    '''
    
    batch_size_total = batch_size * seq_length
    # total number of batches we can make
    n_batches = len(arr)//batch_size_total
    
    # Keep only enough characters to make full batches
    arr = arr[:n_batches * batch_size_total]
    # Reshape into batch_size rows
    arr = arr.reshape((batch_size, -1))
    
    # iterate through the array, one sequence at a time
    for n in range(0, arr.shape[1], seq_length):
        # The features
        x = arr[:, n:n+seq_length]
        # The targets, shifted by one
        y = np.zeros_like(x)
        try:
            y[:, :-1], y[:, -1] = x[:, 1:], arr[:, n+seq_length]
        except IndexError:
            y[:, :-1], y[:, -1] = x[:, 1:], arr[:, 0]
        yield x, y
        
batches = get_batches(encoded, 8, 50)
x, y = next(batches)
# printing out the first 10 items in a sequence
print('x\n', x[:10, :10])
print('\ny\n', y[:10, :10])
"""
x
 [[ 33  33  44  33  33  44  33  33  18  36]
 [ 36  33  27  32  21  11  91  34  99  44]
 [ 68   6  64  28  33  33  33  33  37  32]
 [ 11   8  73  44  37  11  44 100  36  26]
 [ 36  34  44  91  36  34  21  73  33  26]
 [ 73  21  37  11  50  24  19  33  26   4]
 [ 11  10  55  63  34  36  77  44  73  32]
 [ 44  73  32  34  44  48   8  36  50  82]]

y
 [[ 33  44  33  33  44  33  33  18  36  34]
 [ 33  27  32  21  11  91  34  99  44  75]
 [  6  64  28  33  33  33  33  37  32  82]
 [  8  73  44  37  11  44 100  36  26  11]
 [ 34  44  91  36  34  21  73  33  26  21]
 [ 21  37  11  50  24  19  33  26   4  55]
 [ 10  55  63  34  36  77  44  73  32  34]
 [ 73  32  34  44  48   8  36  50  82  99]]
"""

# checking if GPU is available
train_on_gpu = torch.cuda.is_available()
if(train_on_gpu):
    print('Training on GPU!')
else: 
    print('No GPU available, training on CPU; consider making n_epochs very small.')
    
class CharRNN(nn.Module):
    
    def __init__(self, tokens, n_hidden=256, n_layers=2,
                               drop_prob=0.5, lr=0.001):
        super().__init__()
        self.drop_prob = drop_prob
        self.n_layers = n_layers
        self.n_hidden = n_hidden
        self.lr = lr
        
        # creating character dictionaries
        self.chars = tokens
        self.int2char = dict(enumerate(self.chars))
        self.char2int = {ch: ii for ii, ch in self.int2char.items()}
        
        ## TODO: define the layers of the model
        self.lstm = nn.LSTM(len(self.chars), n_hidden, n_layers, dropout=drop_prob, batch_first=True)
        self.dropout = nn.Dropout(drop_prob)
        
        self.fc = nn.Linear(n_hidden, len(self.chars))
    def forward(self, x, hidden):
        ''' Forward pass through the network. 
            These inputs are x, and the hidden/cell state `hidden`. '''
                
        ## TODO: Get the outputs and the new hidden state from the lstm
        r_output, hidden = self.lstm(x, hidden)
        out = self.dropout(r_output)
        out = out.contiguous().view(-1, self.n_hidden)
        
        ## TODO: put x through the fully-connected layer
        out = self.fc(out)
        
        # return the final output and the hidden state
        return out, hidden
    
    
    def init_hidden(self, batch_size):
        ''' Initializes hidden state '''
        # Create two new tensors with sizes n_layers x batch_size x n_hidden,
        # initialized to zero, for hidden state and cell state of LSTM
        weight = next(self.parameters()).data
        
        if (train_on_gpu):
            hidden = (weight.new(self.n_layers, batch_size, self.n_hidden).zero_().cuda(),
                  weight.new(self.n_layers, batch_size, self.n_hidden).zero_().cuda())
        else:
            hidden = (weight.new(self.n_layers, batch_size, self.n_hidden).zero_(),
                      weight.new(self.n_layers, batch_size, self.n_hidden).zero_())
        
        return hidden

    
def train(net, data, epochs=10, batch_size=10, seq_length=50, lr=0.001, clip=5, val_frac=0.1, print_every=10):
    ''' Training a network 
    
        Arguments
        ---------
        
        net: CharRNN network
        data: text data to train the network
        epochs: Number of epochs to train
        batch_size: Number of mini-sequences per mini-batch, aka batch size
        seq_length: Number of character steps per mini-batch
        lr: learning rate
        clip: gradient clipping
        val_frac: Fraction of data to hold out for validation
        print_every: Number of steps for printing training and validation loss
    
    '''
    net.train()
    
    opt = torch.optim.Adam(net.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()
    
    # create training and validation data
    val_idx = int(len(data)*(1-val_frac))
    data, val_data = data[:val_idx], data[val_idx:]
    
    if(train_on_gpu):
        net.cuda()
    
    counter = 0
    n_chars = len(net.chars)
    for e in range(epochs):
        # initialize hidden state
        h = net.init_hidden(batch_size)
        
        for x, y in get_batches(data, batch_size, seq_length):
            counter += 1
            
            # One-hot encode our data and make them Torch tensors
            x = one_hot_encode(x, n_chars)
            inputs, targets = torch.from_numpy(x), torch.from_numpy(y)
            
            if(train_on_gpu):
                inputs, targets = inputs.cuda(), targets.cuda()

            # Creating new variables for the hidden state, otherwise
            # we'd backprop through the entire training history
            h = tuple([each.data for each in h])

            # zero accumulated gradients
            net.zero_grad()
            
            # get the output from the model
            output, h = net(inputs, h)
            
            # calculate the loss and perform backprop
            loss = criterion(output, targets.view(batch_size*seq_length).long())
            loss.backward()
            # `clip_grad_norm` helps prevent the exploding gradient problem in RNNs / LSTMs.
            nn.utils.clip_grad_norm_(net.parameters(), clip)
            opt.step()
            
            # loss stats
            if counter % print_every == 0:
                # Get validation loss
                val_h = net.init_hidden(batch_size)
                val_losses = []
                net.eval()
                for x, y in get_batches(val_data, batch_size, seq_length):
                    # One-hot encode our data and make them Torch tensors
                    x = one_hot_encode(x, n_chars)
                    x, y = torch.from_numpy(x), torch.from_numpy(y)
                    
                    # Creating new variables for the hidden state, otherwise
                    # we'd backprop through the entire training history
                    val_h = tuple([each.data for each in val_h])
                    
                    inputs, targets = x, y
                    if(train_on_gpu):
                        inputs, targets = inputs.cuda(), targets.cuda()

                    output, val_h = net(inputs, val_h)
                    val_loss = criterion(output, targets.view(batch_size*seq_length).long())
                
                    val_losses.append(val_loss.item())
                
                net.train() # reset to train mode after iterationg through validation data
                
                print("Epoch: {}/{}...".format(e+1, epochs),
                      "Step: {}...".format(counter),
                      "Loss: {:.4f}...".format(loss.item()),
                      "Val Loss: {:.4f}".format(np.mean(val_losses)))
                


# defining and printing the net
n_hidden=512
n_layers=2

net = CharRNN(chars, n_hidden, n_layers)
print(net)
"""
CharRNN(
  (lstm): LSTM(105, 512, num_layers=2, batch_first=True, dropout=0.5)
  (dropout): Dropout(p=0.5)
  (fc): Linear(in_features=512, out_features=105, bias=True)
)
"""
#setting the hypyperparameters
batch_size = 128
seq_length = 100
n_epochs = 50  # start small if you are just testing initial behavior

# train the model
train(net, encoded, epochs=n_epochs, batch_size=batch_size, seq_length=seq_length, lr=0.001, print_every=10)
"""
Epoch: 1/50... Step: 10... Loss: 3.3814... Val Loss: 3.3841
Epoch: 1/50... Step: 20... Loss: 3.3045... Val Loss: 3.3133
Epoch: 1/50... Step: 30... Loss: 3.2487... Val Loss: 3.3066
Epoch: 1/50... Step: 40... Loss: 3.2494... Val Loss: 3.2958
Epoch: 1/50... Step: 50... Loss: 3.2625... Val Loss: 3.2935
Epoch: 1/50... Step: 60... Loss: 3.2647... Val Loss: 3.2929
Epoch: 1/50... Step: 70... Loss: 3.2272... Val Loss: 3.2823
Epoch: 1/50... Step: 80... Loss: 3.2354... Val Loss: 3.2978
Epoch: 1/50... Step: 90... Loss: 3.2121... Val Loss: 3.2650
Epoch: 1/50... Step: 100... Loss: 3.2120... Val Loss: 3.2497
Epoch: 1/50... Step: 110... Loss: 3.1694... Val Loss: 3.2033
Epoch: 2/50... Step: 120... Loss: 3.1992... Val Loss: 3.2602
Epoch: 2/50... Step: 130... Loss: 3.1434... Val Loss: 3.1506
Epoch: 2/50... Step: 140... Loss: 2.9850... Val Loss: 3.0194
Epoch: 2/50... Step: 150... Loss: 2.9551... Val Loss: 2.9552
Epoch: 2/50... Step: 160... Loss: 2.8315... Val Loss: 2.8156
Epoch: 2/50... Step: 170... Loss: 2.7305... Val Loss: 2.7389
Epoch: 2/50... Step: 180... Loss: 2.7089... Val Loss: 2.6913
Epoch: 2/50... Step: 190... Loss: 2.6400... Val Loss: 2.6391
Epoch: 2/50... Step: 200... Loss: 2.5940... Val Loss: 2.5746
Epoch: 2/50... Step: 210... Loss: 2.5820... Val Loss: 2.5237
Epoch: 2/50... Step: 220... Loss: 2.5123... Val Loss: 2.4859
Epoch: 3/50... Step: 230... Loss: 2.4484... Val Loss: 2.5213
Epoch: 3/50... Step: 240... Loss: 2.4832... Val Loss: 2.4748
Epoch: 3/50... Step: 250... Loss: 2.4381... Val Loss: 2.3753
Epoch: 3/50... Step: 260... Loss: 2.4464... Val Loss: 2.3355
Epoch: 3/50... Step: 270... Loss: 2.3484... Val Loss: 2.2911
Epoch: 3/50... Step: 280... Loss: 2.3342... Val Loss: 2.2513
Epoch: 3/50... Step: 290... Loss: 2.2466... Val Loss: 2.2103
Epoch: 3/50... Step: 300... Loss: 2.2424... Val Loss: 2.1791
Epoch: 3/50... Step: 310... Loss: 2.2228... Val Loss: 2.1555
Epoch: 3/50... Step: 320... Loss: 2.2275... Val Loss: 2.1149
Epoch: 3/50... Step: 330... Loss: 2.1863... Val Loss: 2.0866
Epoch: 4/50... Step: 340... Loss: 2.1612... Val Loss: 2.0630
Epoch: 4/50... Step: 350... Loss: 2.0858... Val Loss: 2.0344
Epoch: 4/50... Step: 360... Loss: 2.1392... Val Loss: 2.0102
Epoch: 4/50... Step: 370... Loss: 2.1243... Val Loss: 1.9897
Epoch: 4/50... Step: 380... Loss: 2.0926... Val Loss: 1.9648
Epoch: 4/50... Step: 390... Loss: 2.0724... Val Loss: 1.9446
Epoch: 4/50... Step: 400... Loss: 2.0269... Val Loss: 1.9221
Epoch: 4/50... Step: 410... Loss: 2.0433... Val Loss: 1.9051
Epoch: 4/50... Step: 420... Loss: 2.0029... Val Loss: 1.8885
Epoch: 4/50... Step: 430... Loss: 1.9725... Val Loss: 1.8736
Epoch: 4/50... Step: 440... Loss: 2.0055... Val Loss: 1.8465
Epoch: 5/50... Step: 450... Loss: 1.9516... Val Loss: 1.8313
Epoch: 5/50... Step: 460... Loss: 1.9585... Val Loss: 1.8154
Epoch: 5/50... Step: 470... Loss: 1.9323... Val Loss: 1.7963
Epoch: 5/50... Step: 480... Loss: 1.8983... Val Loss: 1.7757
Epoch: 5/50... Step: 490... Loss: 1.8877... Val Loss: 1.7654
Epoch: 5/50... Step: 500... Loss: 1.9041... Val Loss: 1.7514
Epoch: 5/50... Step: 510... Loss: 1.9082... Val Loss: 1.7388
Epoch: 5/50... Step: 520... Loss: 1.8657... Val Loss: 1.7225
Epoch: 5/50... Step: 530... Loss: 1.8635... Val Loss: 1.7114
Epoch: 5/50... Step: 540... Loss: 1.8731... Val Loss: 1.6994
Epoch: 5/50... Step: 550... Loss: 1.8021... Val Loss: 1.6840
Epoch: 5/50... Step: 560... Loss: 1.8169... Val Loss: 1.6687
Epoch: 6/50... Step: 570... Loss: 1.8338... Val Loss: 1.6547
Epoch: 6/50... Step: 580... Loss: 1.8004... Val Loss: 1.6465
Epoch: 6/50... Step: 590... Loss: 1.7841... Val Loss: 1.6377
Epoch: 6/50... Step: 600... Loss: 1.7635... Val Loss: 1.6251
Epoch: 6/50... Step: 610... Loss: 1.8102... Val Loss: 1.6176
Epoch: 6/50... Step: 620... Loss: 1.7207... Val Loss: 1.6069
Epoch: 6/50... Step: 630... Loss: 1.7206... Val Loss: 1.5946
Epoch: 6/50... Step: 640... Loss: 1.7395... Val Loss: 1.5815
Epoch: 6/50... Step: 650... Loss: 1.7063... Val Loss: 1.5777
Epoch: 6/50... Step: 660... Loss: 1.6879... Val Loss: 1.5666
Epoch: 6/50... Step: 670... Loss: 1.7183... Val Loss: 1.5561
Epoch: 7/50... Step: 680... Loss: 1.6549... Val Loss: 1.5449
Epoch: 7/50... Step: 690... Loss: 1.6483... Val Loss: 1.5385
Epoch: 7/50... Step: 700... Loss: 1.6781... Val Loss: 1.5315
Epoch: 7/50... Step: 710... Loss: 1.6960... Val Loss: 1.5214
Epoch: 7/50... Step: 720... Loss: 1.7075... Val Loss: 1.5172
Epoch: 7/50... Step: 730... Loss: 1.6484... Val Loss: 1.5065
Epoch: 7/50... Step: 740... Loss: 1.5929... Val Loss: 1.4986
Epoch: 7/50... Step: 750... Loss: 1.6137... Val Loss: 1.4932
Epoch: 7/50... Step: 760... Loss: 1.6549... Val Loss: 1.4814
Epoch: 7/50... Step: 770... Loss: 1.6346... Val Loss: 1.4772
Epoch: 7/50... Step: 780... Loss: 1.6368... Val Loss: 1.4697
Epoch: 8/50... Step: 790... Loss: 1.5723... Val Loss: 1.4664
Epoch: 8/50... Step: 800... Loss: 1.5784... Val Loss: 1.4566
Epoch: 8/50... Step: 810... Loss: 1.5789... Val Loss: 1.4518
Epoch: 8/50... Step: 820... Loss: 1.6377... Val Loss: 1.4457
Epoch: 8/50... Step: 830... Loss: 1.5776... Val Loss: 1.4394
Epoch: 8/50... Step: 840... Loss: 1.5760... Val Loss: 1.4361
Epoch: 8/50... Step: 850... Loss: 1.5114... Val Loss: 1.4255
Epoch: 8/50... Step: 860... Loss: 1.5150... Val Loss: 1.4207
Epoch: 8/50... Step: 870... Loss: 1.5276... Val Loss: 1.4144
Epoch: 8/50... Step: 880... Loss: 1.5601... Val Loss: 1.4108
Epoch: 8/50... Step: 890... Loss: 1.5287... Val Loss: 1.4054
Epoch: 9/50... Step: 900... Loss: 1.5476... Val Loss: 1.3978
Epoch: 9/50... Step: 910... Loss: 1.4725... Val Loss: 1.3927
Epoch: 9/50... Step: 920... Loss: 1.5840... Val Loss: 1.3899
Epoch: 9/50... Step: 930... Loss: 1.5339... Val Loss: 1.3834
Epoch: 9/50... Step: 940... Loss: 1.5202... Val Loss: 1.3831
Epoch: 9/50... Step: 950... Loss: 1.5087... Val Loss: 1.3788
Epoch: 9/50... Step: 960... Loss: 1.5120... Val Loss: 1.3697
Epoch: 9/50... Step: 970... Loss: 1.5172... Val Loss: 1.3670
Epoch: 9/50... Step: 980... Loss: 1.4957... Val Loss: 1.3605
Epoch: 9/50... Step: 990... Loss: 1.4753... Val Loss: 1.3579
Epoch: 9/50... Step: 1000... Loss: 1.5355... Val Loss: 1.3518
Epoch: 10/50... Step: 1010... Loss: 1.4745... Val Loss: 1.3451
Epoch: 10/50... Step: 1020... Loss: 1.4997... Val Loss: 1.3474
Epoch: 10/50... Step: 1030... Loss: 1.4894... Val Loss: 1.3400
Epoch: 10/50... Step: 1040... Loss: 1.4401... Val Loss: 1.3366
Epoch: 10/50... Step: 1050... Loss: 1.4730... Val Loss: 1.3345
Epoch: 10/50... Step: 1060... Loss: 1.4960... Val Loss: 1.3311
Epoch: 10/50... Step: 1070... Loss: 1.5027... Val Loss: 1.3269
Epoch: 10/50... Step: 1080... Loss: 1.4663... Val Loss: 1.3214
Epoch: 10/50... Step: 1090... Loss: 1.4775... Val Loss: 1.3170
Epoch: 10/50... Step: 1100... Loss: 1.5033... Val Loss: 1.3163
Epoch: 10/50... Step: 1110... Loss: 1.4124... Val Loss: 1.3102
Epoch: 10/50... Step: 1120... Loss: 1.4595... Val Loss: 1.3036
Epoch: 11/50... Step: 1130... Loss: 1.4701... Val Loss: 1.3033
Epoch: 11/50... Step: 1140... Loss: 1.4389... Val Loss: 1.2996
Epoch: 11/50... Step: 1150... Loss: 1.4279... Val Loss: 1.2993
Epoch: 11/50... Step: 1160... Loss: 1.4260... Val Loss: 1.2942
Epoch: 11/50... Step: 1170... Loss: 1.4857... Val Loss: 1.2935
Epoch: 11/50... Step: 1180... Loss: 1.3946... Val Loss: 1.2898
Epoch: 11/50... Step: 1190... Loss: 1.4204... Val Loss: 1.2844
Epoch: 11/50... Step: 1200... Loss: 1.4266... Val Loss: 1.2809
Epoch: 11/50... Step: 1210... Loss: 1.4124... Val Loss: 1.2782
Epoch: 11/50... Step: 1220... Loss: 1.3871... Val Loss: 1.2752
Epoch: 11/50... Step: 1230... Loss: 1.4091... Val Loss: 1.2702
Epoch: 12/50... Step: 1240... Loss: 1.3614... Val Loss: 1.2695
Epoch: 12/50... Step: 1250... Loss: 1.3705... Val Loss: 1.2681
Epoch: 12/50... Step: 1260... Loss: 1.3729... Val Loss: 1.2669
Epoch: 12/50... Step: 1270... Loss: 1.4248... Val Loss: 1.2610
Epoch: 12/50... Step: 1280... Loss: 1.4407... Val Loss: 1.2592
Epoch: 12/50... Step: 1290... Loss: 1.3787... Val Loss: 1.2601
Epoch: 12/50... Step: 1300... Loss: 1.3322... Val Loss: 1.2541
Epoch: 12/50... Step: 1310... Loss: 1.3503... Val Loss: 1.2534
Epoch: 12/50... Step: 1320... Loss: 1.4036... Val Loss: 1.2507
Epoch: 12/50... Step: 1330... Loss: 1.3905... Val Loss: 1.2485
Epoch: 12/50... Step: 1340... Loss: 1.3770... Val Loss: 1.2433
Epoch: 13/50... Step: 1350... Loss: 1.3404... Val Loss: 1.2400
Epoch: 13/50... Step: 1360... Loss: 1.3438... Val Loss: 1.2388
Epoch: 13/50... Step: 1370... Loss: 1.3489... Val Loss: 1.2397
Epoch: 13/50... Step: 1380... Loss: 1.4004... Val Loss: 1.2350
Epoch: 13/50... Step: 1390... Loss: 1.3472... Val Loss: 1.2351
Epoch: 13/50... Step: 1400... Loss: 1.3673... Val Loss: 1.2334
Epoch: 13/50... Step: 1410... Loss: 1.2930... Val Loss: 1.2294
Epoch: 13/50... Step: 1420... Loss: 1.2859... Val Loss: 1.2288
Epoch: 13/50... Step: 1430... Loss: 1.3155... Val Loss: 1.2251
Epoch: 13/50... Step: 1440... Loss: 1.3533... Val Loss: 1.2221
Epoch: 13/50... Step: 1450... Loss: 1.3183... Val Loss: 1.2199
Epoch: 14/50... Step: 1460... Loss: 1.3348... Val Loss: 1.2179
Epoch: 14/50... Step: 1470... Loss: 1.2723... Val Loss: 1.2204
Epoch: 14/50... Step: 1480... Loss: 1.3851... Val Loss: 1.2166
Epoch: 14/50... Step: 1490... Loss: 1.3418... Val Loss: 1.2134
Epoch: 14/50... Step: 1500... Loss: 1.3255... Val Loss: 1.2131
Epoch: 14/50... Step: 1510... Loss: 1.3224... Val Loss: 1.2089
Epoch: 14/50... Step: 1520... Loss: 1.3178... Val Loss: 1.2090
Epoch: 14/50... Step: 1530... Loss: 1.3220... Val Loss: 1.2060
Epoch: 14/50... Step: 1540... Loss: 1.3137... Val Loss: 1.2028
Epoch: 14/50... Step: 1550... Loss: 1.3048... Val Loss: 1.2011
Epoch: 14/50... Step: 1560... Loss: 1.3463... Val Loss: 1.1979
Epoch: 15/50... Step: 1570... Loss: 1.2979... Val Loss: 1.1972
Epoch: 15/50... Step: 1580... Loss: 1.3155... Val Loss: 1.1979
Epoch: 15/50... Step: 1590... Loss: 1.3155... Val Loss: 1.1953
Epoch: 15/50... Step: 1600... Loss: 1.2773... Val Loss: 1.1963
Epoch: 15/50... Step: 1610... Loss: 1.3118... Val Loss: 1.1929
Epoch: 15/50... Step: 1620... Loss: 1.3296... Val Loss: 1.1901
Epoch: 15/50... Step: 1630... Loss: 1.3345... Val Loss: 1.1921
Epoch: 15/50... Step: 1640... Loss: 1.3169... Val Loss: 1.1892
Epoch: 15/50... Step: 1650... Loss: 1.3094... Val Loss: 1.1868
Epoch: 15/50... Step: 1660... Loss: 1.3358... Val Loss: 1.1844
Epoch: 15/50... Step: 1670... Loss: 1.2685... Val Loss: 1.1822
Epoch: 15/50... Step: 1680... Loss: 1.3134... Val Loss: 1.1792
Epoch: 16/50... Step: 1690... Loss: 1.3054... Val Loss: 1.1779
Epoch: 16/50... Step: 1700... Loss: 1.2837... Val Loss: 1.1777
Epoch: 16/50... Step: 1710... Loss: 1.2655... Val Loss: 1.1807
Epoch: 16/50... Step: 1720... Loss: 1.2779... Val Loss: 1.1763
Epoch: 16/50... Step: 1730... Loss: 1.3366... Val Loss: 1.1756
Epoch: 16/50... Step: 1740... Loss: 1.2533... Val Loss: 1.1759
Epoch: 16/50... Step: 1750... Loss: 1.2800... Val Loss: 1.1725
Epoch: 16/50... Step: 1760... Loss: 1.2833... Val Loss: 1.1702
Epoch: 16/50... Step: 1770... Loss: 1.2581... Val Loss: 1.1685
Epoch: 16/50... Step: 1780... Loss: 1.2489... Val Loss: 1.1693
Epoch: 16/50... Step: 1790... Loss: 1.2716... Val Loss: 1.1654
Epoch: 17/50... Step: 1800... Loss: 1.2278... Val Loss: 1.1659
Epoch: 17/50... Step: 1810... Loss: 1.2437... Val Loss: 1.1641
Epoch: 17/50... Step: 1820... Loss: 1.2439... Val Loss: 1.1661
Epoch: 17/50... Step: 1830... Loss: 1.2769... Val Loss: 1.1616
Epoch: 17/50... Step: 1840... Loss: 1.3020... Val Loss: 1.1616
Epoch: 17/50... Step: 1850... Loss: 1.2565... Val Loss: 1.1595
Epoch: 17/50... Step: 1860... Loss: 1.2130... Val Loss: 1.1574
Epoch: 17/50... Step: 1870... Loss: 1.2252... Val Loss: 1.1558
Epoch: 17/50... Step: 1880... Loss: 1.2763... Val Loss: 1.1538
Epoch: 17/50... Step: 1890... Loss: 1.2736... Val Loss: 1.1529
Epoch: 17/50... Step: 1900... Loss: 1.2502... Val Loss: 1.1531
Epoch: 18/50... Step: 1910... Loss: 1.2231... Val Loss: 1.1506
Epoch: 18/50... Step: 1920... Loss: 1.2237... Val Loss: 1.1528
Epoch: 18/50... Step: 1930... Loss: 1.2327... Val Loss: 1.1523
Epoch: 18/50... Step: 1940... Loss: 1.2844... Val Loss: 1.1490
Epoch: 18/50... Step: 1950... Loss: 1.2339... Val Loss: 1.1474
Epoch: 18/50... Step: 1960... Loss: 1.2474... Val Loss: 1.1468
Epoch: 18/50... Step: 1970... Loss: 1.1857... Val Loss: 1.1454
Epoch: 18/50... Step: 1980... Loss: 1.1754... Val Loss: 1.1430
Epoch: 18/50... Step: 1990... Loss: 1.1919... Val Loss: 1.1426
Epoch: 18/50... Step: 2000... Loss: 1.2302... Val Loss: 1.1401
Epoch: 18/50... Step: 2010... Loss: 1.2101... Val Loss: 1.1380
Epoch: 19/50... Step: 2020... Loss: 1.2293... Val Loss: 1.1372
Epoch: 19/50... Step: 2030... Loss: 1.1640... Val Loss: 1.1396
Epoch: 19/50... Step: 2040... Loss: 1.2715... Val Loss: 1.1395
Epoch: 19/50... Step: 2050... Loss: 1.2381... Val Loss: 1.1371
Epoch: 19/50... Step: 2060... Loss: 1.2302... Val Loss: 1.1351
Epoch: 19/50... Step: 2070... Loss: 1.2005... Val Loss: 1.1331
Epoch: 19/50... Step: 2080... Loss: 1.2046... Val Loss: 1.1349
Epoch: 19/50... Step: 2090... Loss: 1.2133... Val Loss: 1.1327
Epoch: 19/50... Step: 2100... Loss: 1.2137... Val Loss: 1.1322
Epoch: 19/50... Step: 2110... Loss: 1.2048... Val Loss: 1.1315
Epoch: 19/50... Step: 2120... Loss: 1.2482... Val Loss: 1.1297
Epoch: 20/50... Step: 2130... Loss: 1.1983... Val Loss: 1.1266
Epoch: 20/50... Step: 2140... Loss: 1.2203... Val Loss: 1.1292
Epoch: 20/50... Step: 2150... Loss: 1.2220... Val Loss: 1.1266
Epoch: 20/50... Step: 2160... Loss: 1.1767... Val Loss: 1.1279
Epoch: 20/50... Step: 2170... Loss: 1.2219... Val Loss: 1.1260
Epoch: 20/50... Step: 2180... Loss: 1.2206... Val Loss: 1.1226
Epoch: 20/50... Step: 2190... Loss: 1.2438... Val Loss: 1.1260
Epoch: 20/50... Step: 2200... Loss: 1.2051... Val Loss: 1.1234
Epoch: 20/50... Step: 2210... Loss: 1.2164... Val Loss: 1.1216
Epoch: 20/50... Step: 2220... Loss: 1.2518... Val Loss: 1.1192
Epoch: 20/50... Step: 2230... Loss: 1.1689... Val Loss: 1.1223
Epoch: 20/50... Step: 2240... Loss: 1.2169... Val Loss: 1.1174
Epoch: 21/50... Step: 2250... Loss: 1.2044... Val Loss: 1.1187
Epoch: 21/50... Step: 2260... Loss: 1.1870... Val Loss: 1.1188
Epoch: 21/50... Step: 2270... Loss: 1.1780... Val Loss: 1.1197
Epoch: 21/50... Step: 2280... Loss: 1.1739... Val Loss: 1.1162
Epoch: 21/50... Step: 2290... Loss: 1.2372... Val Loss: 1.1161
Epoch: 21/50... Step: 2300... Loss: 1.1749... Val Loss: 1.1174
Epoch: 21/50... Step: 2310... Loss: 1.1861... Val Loss: 1.1158
Epoch: 21/50... Step: 2320... Loss: 1.1954... Val Loss: 1.1116
Epoch: 21/50... Step: 2330... Loss: 1.1685... Val Loss: 1.1126
Epoch: 21/50... Step: 2340... Loss: 1.1742... Val Loss: 1.1113
Epoch: 21/50... Step: 2350... Loss: 1.1802... Val Loss: 1.1107
Epoch: 22/50... Step: 2360... Loss: 1.1409... Val Loss: 1.1113
Epoch: 22/50... Step: 2370... Loss: 1.1544... Val Loss: 1.1143
Epoch: 22/50... Step: 2380... Loss: 1.1563... Val Loss: 1.1110
Epoch: 22/50... Step: 2390... Loss: 1.2053... Val Loss: 1.1069
Epoch: 22/50... Step: 2400... Loss: 1.2156... Val Loss: 1.1071
Epoch: 22/50... Step: 2410... Loss: 1.1858... Val Loss: 1.1051
Epoch: 22/50... Step: 2420... Loss: 1.1348... Val Loss: 1.1062
Epoch: 22/50... Step: 2430... Loss: 1.1407... Val Loss: 1.1045
Epoch: 22/50... Step: 2440... Loss: 1.1960... Val Loss: 1.1030
Epoch: 22/50... Step: 2450... Loss: 1.1943... Val Loss: 1.1033
Epoch: 22/50... Step: 2460... Loss: 1.1731... Val Loss: 1.1010
Epoch: 23/50... Step: 2470... Loss: 1.1399... Val Loss: 1.1003
Epoch: 23/50... Step: 2480... Loss: 1.1525... Val Loss: 1.1028
Epoch: 23/50... Step: 2490... Loss: 1.1514... Val Loss: 1.1031
Epoch: 23/50... Step: 2500... Loss: 1.2014... Val Loss: 1.1027
Epoch: 23/50... Step: 2510... Loss: 1.1537... Val Loss: 1.1009
Epoch: 23/50... Step: 2520... Loss: 1.1761... Val Loss: 1.0988
Epoch: 23/50... Step: 2530... Loss: 1.1080... Val Loss: 1.0986
Epoch: 23/50... Step: 2540... Loss: 1.1013... Val Loss: 1.0986
Epoch: 23/50... Step: 2550... Loss: 1.1140... Val Loss: 1.0969
Epoch: 23/50... Step: 2560... Loss: 1.1635... Val Loss: 1.0958
Epoch: 23/50... Step: 2570... Loss: 1.1369... Val Loss: 1.0964
Epoch: 24/50... Step: 2580... Loss: 1.1609... Val Loss: 1.0934
Epoch: 24/50... Step: 2590... Loss: 1.0983... Val Loss: 1.0962
Epoch: 24/50... Step: 2600... Loss: 1.1893... Val Loss: 1.0949
Epoch: 24/50... Step: 2610... Loss: 1.1554... Val Loss: 1.0960
Epoch: 24/50... Step: 2620... Loss: 1.1566... Val Loss: 1.0952
Epoch: 24/50... Step: 2630... Loss: 1.1399... Val Loss: 1.0925
Epoch: 24/50... Step: 2640... Loss: 1.1461... Val Loss: 1.0945
Epoch: 24/50... Step: 2650... Loss: 1.1475... Val Loss: 1.0906
Epoch: 24/50... Step: 2660... Loss: 1.1344... Val Loss: 1.0917
Epoch: 24/50... Step: 2670... Loss: 1.1327... Val Loss: 1.0893
Epoch: 24/50... Step: 2680... Loss: 1.1717... Val Loss: 1.0888
Epoch: 25/50... Step: 2690... Loss: 1.1259... Val Loss: 1.0876
Epoch: 25/50... Step: 2700... Loss: 1.1482... Val Loss: 1.0904
Epoch: 25/50... Step: 2710... Loss: 1.1496... Val Loss: 1.0887
Epoch: 25/50... Step: 2720... Loss: 1.1099... Val Loss: 1.0877
Epoch: 25/50... Step: 2730... Loss: 1.1518... Val Loss: 1.0879
Epoch: 25/50... Step: 2740... Loss: 1.1585... Val Loss: 1.0866
Epoch: 25/50... Step: 2750... Loss: 1.1677... Val Loss: 1.0865
Epoch: 25/50... Step: 2760... Loss: 1.1532... Val Loss: 1.0867
Epoch: 25/50... Step: 2770... Loss: 1.1512... Val Loss: 1.0856
Epoch: 25/50... Step: 2780... Loss: 1.1780... Val Loss: 1.0834
Epoch: 25/50... Step: 2790... Loss: 1.1047... Val Loss: 1.0846
Epoch: 25/50... Step: 2800... Loss: 1.1646... Val Loss: 1.0822
Epoch: 26/50... Step: 2810... Loss: 1.1443... Val Loss: 1.0809
Epoch: 26/50... Step: 2820... Loss: 1.1266... Val Loss: 1.0863
Epoch: 26/50... Step: 2830... Loss: 1.1059... Val Loss: 1.0868
Epoch: 26/50... Step: 2840... Loss: 1.1095... Val Loss: 1.0805
Epoch: 26/50... Step: 2850... Loss: 1.1714... Val Loss: 1.0807
Epoch: 26/50... Step: 2860... Loss: 1.1150... Val Loss: 1.0813
Epoch: 26/50... Step: 2870... Loss: 1.1307... Val Loss: 1.0815
Epoch: 26/50... Step: 2880... Loss: 1.1333... Val Loss: 1.0810
Epoch: 26/50... Step: 2890... Loss: 1.1024... Val Loss: 1.0799
Epoch: 26/50... Step: 2900... Loss: 1.0966... Val Loss: 1.0805
Epoch: 26/50... Step: 2910... Loss: 1.1247... Val Loss: 1.0778
Epoch: 27/50... Step: 2920... Loss: 1.0797... Val Loss: 1.0767
Epoch: 27/50... Step: 2930... Loss: 1.0987... Val Loss: 1.0820
Epoch: 27/50... Step: 2940... Loss: 1.0904... Val Loss: 1.0796
Epoch: 27/50... Step: 2950... Loss: 1.1318... Val Loss: 1.0786
Epoch: 27/50... Step: 2960... Loss: 1.1634... Val Loss: 1.0781
Epoch: 27/50... Step: 2970... Loss: 1.1192... Val Loss: 1.0776
Epoch: 27/50... Step: 2980... Loss: 1.0741... Val Loss: 1.0783
Epoch: 27/50... Step: 2990... Loss: 1.0875... Val Loss: 1.0746
Epoch: 27/50... Step: 3000... Loss: 1.1290... Val Loss: 1.0737
Epoch: 27/50... Step: 3010... Loss: 1.1415... Val Loss: 1.0753
Epoch: 27/50... Step: 3020... Loss: 1.1067... Val Loss: 1.0746
Epoch: 28/50... Step: 3030... Loss: 1.0917... Val Loss: 1.0693
Epoch: 28/50... Step: 3040... Loss: 1.0991... Val Loss: 1.0746
Epoch: 28/50... Step: 3050... Loss: 1.0886... Val Loss: 1.0732
Epoch: 28/50... Step: 3060... Loss: 1.1386... Val Loss: 1.0705
Epoch: 28/50... Step: 3070... Loss: 1.1070... Val Loss: 1.0714
Epoch: 28/50... Step: 3080... Loss: 1.1191... Val Loss: 1.0689
Epoch: 28/50... Step: 3090... Loss: 1.0586... Val Loss: 1.0705
Epoch: 28/50... Step: 3100... Loss: 1.0518... Val Loss: 1.0666
Epoch: 28/50... Step: 3110... Loss: 1.0582... Val Loss: 1.0678
Epoch: 28/50... Step: 3120... Loss: 1.0956... Val Loss: 1.0639
Epoch: 28/50... Step: 3130... Loss: 1.0742... Val Loss: 1.0664
Epoch: 29/50... Step: 3140... Loss: 1.1025... Val Loss: 1.0625
Epoch: 29/50... Step: 3150... Loss: 1.0356... Val Loss: 1.0643
Epoch: 29/50... Step: 3160... Loss: 1.1373... Val Loss: 1.0627
Epoch: 29/50... Step: 3170... Loss: 1.0968... Val Loss: 1.0642
Epoch: 29/50... Step: 3180... Loss: 1.1015... Val Loss: 1.0623
Epoch: 29/50... Step: 3190... Loss: 1.0788... Val Loss: 1.0585
Epoch: 29/50... Step: 3200... Loss: 1.0693... Val Loss: 1.0598
Epoch: 29/50... Step: 3210... Loss: 1.1012... Val Loss: 1.0590
Epoch: 29/50... Step: 3220... Loss: 1.0656... Val Loss: 1.0564
Epoch: 29/50... Step: 3230... Loss: 1.0784... Val Loss: 1.0538
Epoch: 29/50... Step: 3240... Loss: 1.1125... Val Loss: 1.0542
Epoch: 30/50... Step: 3250... Loss: 1.0729... Val Loss: 1.0553
Epoch: 30/50... Step: 3260... Loss: 1.0892... Val Loss: 1.0553
Epoch: 30/50... Step: 3270... Loss: 1.0956... Val Loss: 1.0566
Epoch: 30/50... Step: 3280... Loss: 1.0431... Val Loss: 1.0562
Epoch: 30/50... Step: 3290... Loss: 1.0883... Val Loss: 1.0528
Epoch: 30/50... Step: 3300... Loss: 1.0865... Val Loss: 1.0510
Epoch: 30/50... Step: 3310... Loss: 1.1142... Val Loss: 1.0517
Epoch: 30/50... Step: 3320... Loss: 1.0897... Val Loss: 1.0521
Epoch: 30/50... Step: 3330... Loss: 1.0935... Val Loss: 1.0536
Epoch: 30/50... Step: 3340... Loss: 1.1106... Val Loss: 1.0527
Epoch: 30/50... Step: 3350... Loss: 1.0417... Val Loss: 1.0513
Epoch: 30/50... Step: 3360... Loss: 1.1069... Val Loss: 1.0516
Epoch: 31/50... Step: 3370... Loss: 1.0783... Val Loss: 1.0477
Epoch: 31/50... Step: 3380... Loss: 1.0781... Val Loss: 1.0554
Epoch: 31/50... Step: 3390... Loss: 1.0568... Val Loss: 1.0506
Epoch: 31/50... Step: 3400... Loss: 1.0533... Val Loss: 1.0497
Epoch: 31/50... Step: 3410... Loss: 1.1071... Val Loss: 1.0488
Epoch: 31/50... Step: 3420... Loss: 1.0502... Val Loss: 1.0488
Epoch: 31/50... Step: 3430... Loss: 1.0711... Val Loss: 1.0504
Epoch: 31/50... Step: 3440... Loss: 1.0752... Val Loss: 1.0435
Epoch: 31/50... Step: 3450... Loss: 1.0443... Val Loss: 1.0446
Epoch: 31/50... Step: 3460... Loss: 1.0317... Val Loss: 1.0467
Epoch: 31/50... Step: 3470... Loss: 1.0624... Val Loss: 1.0451
Epoch: 32/50... Step: 3480... Loss: 1.0118... Val Loss: 1.0431
Epoch: 32/50... Step: 3490... Loss: 1.0341... Val Loss: 1.0479
Epoch: 32/50... Step: 3500... Loss: 1.0308... Val Loss: 1.0449
Epoch: 32/50... Step: 3510... Loss: 1.0742... Val Loss: 1.0432
Epoch: 32/50... Step: 3520... Loss: 1.0965... Val Loss: 1.0484
Epoch: 32/50... Step: 3530... Loss: 1.0577... Val Loss: 1.0484
Epoch: 32/50... Step: 3540... Loss: 1.0203... Val Loss: 1.0482
Epoch: 32/50... Step: 3550... Loss: 1.0407... Val Loss: 1.0397
Epoch: 32/50... Step: 3560... Loss: 1.0672... Val Loss: 1.0447
Epoch: 32/50... Step: 3570... Loss: 1.0827... Val Loss: 1.0427
Epoch: 32/50... Step: 3580... Loss: 1.0511... Val Loss: 1.0450
Epoch: 33/50... Step: 3590... Loss: 1.0368... Val Loss: 1.0410
Epoch: 33/50... Step: 3600... Loss: 1.0425... Val Loss: 1.0453
Epoch: 33/50... Step: 3610... Loss: 1.0378... Val Loss: 1.0469
Epoch: 33/50... Step: 3620... Loss: 1.0838... Val Loss: 1.0423
Epoch: 33/50... Step: 3630... Loss: 1.0450... Val Loss: 1.0418
Epoch: 33/50... Step: 3640... Loss: 1.0655... Val Loss: 1.0403
Epoch: 33/50... Step: 3650... Loss: 0.9996... Val Loss: 1.0414
Epoch: 33/50... Step: 3660... Loss: 0.9965... Val Loss: 1.0408
Epoch: 33/50... Step: 3670... Loss: 1.0032... Val Loss: 1.0396
Epoch: 33/50... Step: 3680... Loss: 1.0514... Val Loss: 1.0389
Epoch: 33/50... Step: 3690... Loss: 1.0221... Val Loss: 1.0377
Epoch: 34/50... Step: 3700... Loss: 1.0460... Val Loss: 1.0398
Epoch: 34/50... Step: 3710... Loss: 0.9947... Val Loss: 1.0405
Epoch: 34/50... Step: 3720... Loss: 1.0835... Val Loss: 1.0417
Epoch: 34/50... Step: 3730... Loss: 1.0397... Val Loss: 1.0422
Epoch: 34/50... Step: 3740... Loss: 1.0432... Val Loss: 1.0398
Epoch: 34/50... Step: 3750... Loss: 1.0273... Val Loss: 1.0362
Epoch: 34/50... Step: 3760... Loss: 1.0238... Val Loss: 1.0419
Epoch: 34/50... Step: 3770... Loss: 1.0404... Val Loss: 1.0405
Epoch: 34/50... Step: 3780... Loss: 1.0356... Val Loss: 1.0396
Epoch: 34/50... Step: 3790... Loss: 1.0311... Val Loss: 1.0372
Epoch: 34/50... Step: 3800... Loss: 1.0635... Val Loss: 1.0354
Epoch: 35/50... Step: 3810... Loss: 1.0345... Val Loss: 1.0384
Epoch: 35/50... Step: 3820... Loss: 1.0376... Val Loss: 1.0368
Epoch: 35/50... Step: 3830... Loss: 1.0452... Val Loss: 1.0387
Epoch: 35/50... Step: 3840... Loss: 1.0067... Val Loss: 1.0413
Epoch: 35/50... Step: 3850... Loss: 1.0414... Val Loss: 1.0389
Epoch: 35/50... Step: 3860... Loss: 1.0431... Val Loss: 1.0361
Epoch: 35/50... Step: 3870... Loss: 1.0730... Val Loss: 1.0385
Epoch: 35/50... Step: 3880... Loss: 1.0533... Val Loss: 1.0397
Epoch: 35/50... Step: 3890... Loss: 1.0409... Val Loss: 1.0341
Epoch: 35/50... Step: 3900... Loss: 1.0696... Val Loss: 1.0351
Epoch: 35/50... Step: 3910... Loss: 1.0087... Val Loss: 1.0387
Epoch: 35/50... Step: 3920... Loss: 1.0636... Val Loss: 1.0370
Epoch: 36/50... Step: 3930... Loss: 1.0359... Val Loss: 1.0376
Epoch: 36/50... Step: 3940... Loss: 1.0236... Val Loss: 1.0413
Epoch: 36/50... Step: 3950... Loss: 1.0111... Val Loss: 1.0364
Epoch: 36/50... Step: 3960... Loss: 0.9992... Val Loss: 1.0366
Epoch: 36/50... Step: 3970... Loss: 1.0641... Val Loss: 1.0427
Epoch: 36/50... Step: 3980... Loss: 1.0129... Val Loss: 1.0386
Epoch: 36/50... Step: 3990... Loss: 1.0390... Val Loss: 1.0372
Epoch: 36/50... Step: 4000... Loss: 1.0445... Val Loss: 1.0336
Epoch: 36/50... Step: 4010... Loss: 1.0001... Val Loss: 1.0349
Epoch: 36/50... Step: 4020... Loss: 1.0040... Val Loss: 1.0337
Epoch: 36/50... Step: 4030... Loss: 1.0323... Val Loss: 1.0389
Epoch: 37/50... Step: 4040... Loss: 0.9800... Val Loss: 1.0343
Epoch: 37/50... Step: 4050... Loss: 1.0002... Val Loss: 1.0412
Epoch: 37/50... Step: 4060... Loss: 0.9932... Val Loss: 1.0387
Epoch: 37/50... Step: 4070... Loss: 1.0317... Val Loss: 1.0346
Epoch: 37/50... Step: 4080... Loss: 1.0550... Val Loss: 1.0399
Epoch: 37/50... Step: 4090... Loss: 1.0212... Val Loss: 1.0374
Epoch: 37/50... Step: 4100... Loss: 0.9923... Val Loss: 1.0393
Epoch: 37/50... Step: 4110... Loss: 0.9947... Val Loss: 1.0384
Epoch: 37/50... Step: 4120... Loss: 1.0366... Val Loss: 1.0371
Epoch: 37/50... Step: 4130... Loss: 1.0314... Val Loss: 1.0325
Epoch: 37/50... Step: 4140... Loss: 1.0106... Val Loss: 1.0316
Epoch: 38/50... Step: 4150... Loss: 1.0069... Val Loss: 1.0361
Epoch: 38/50... Step: 4160... Loss: 1.0025... Val Loss: 1.0377
Epoch: 38/50... Step: 4170... Loss: 1.0004... Val Loss: 1.0349
Epoch: 38/50... Step: 4180... Loss: 1.0357... Val Loss: 1.0365
Epoch: 38/50... Step: 4190... Loss: 1.0057... Val Loss: 1.0357
Epoch: 38/50... Step: 4200... Loss: 1.0196... Val Loss: 1.0335
Epoch: 38/50... Step: 4210... Loss: 0.9654... Val Loss: 1.0353
Epoch: 38/50... Step: 4220... Loss: 0.9570... Val Loss: 1.0322
Epoch: 38/50... Step: 4230... Loss: 0.9659... Val Loss: 1.0354
Epoch: 38/50... Step: 4240... Loss: 1.0136... Val Loss: 1.0334
Epoch: 38/50... Step: 4250... Loss: 0.9951... Val Loss: 1.0305
Epoch: 39/50... Step: 4260... Loss: 1.0124... Val Loss: 1.0346
Epoch: 39/50... Step: 4270... Loss: 0.9694... Val Loss: 1.0343
Epoch: 39/50... Step: 4280... Loss: 1.0481... Val Loss: 1.0353
Epoch: 39/50... Step: 4290... Loss: 1.0024... Val Loss: 1.0352
Epoch: 39/50... Step: 4300... Loss: 1.0080... Val Loss: 1.0367
Epoch: 39/50... Step: 4310... Loss: 0.9896... Val Loss: 1.0323
Epoch: 39/50... Step: 4320... Loss: 0.9872... Val Loss: 1.0337
Epoch: 39/50... Step: 4330... Loss: 1.0094... Val Loss: 1.0357
Epoch: 39/50... Step: 4340... Loss: 1.0039... Val Loss: 1.0324
Epoch: 39/50... Step: 4350... Loss: 0.9991... Val Loss: 1.0323
Epoch: 39/50... Step: 4360... Loss: 1.0286... Val Loss: 1.0359
Epoch: 40/50... Step: 4370... Loss: 0.9995... Val Loss: 1.0344
Epoch: 40/50... Step: 4380... Loss: 1.0126... Val Loss: 1.0310
Epoch: 40/50... Step: 4390... Loss: 1.0117... Val Loss: 1.0402
Epoch: 40/50... Step: 4400... Loss: 0.9856... Val Loss: 1.0327
Epoch: 40/50... Step: 4410... Loss: 1.0173... Val Loss: 1.0372
Epoch: 40/50... Step: 4420... Loss: 1.0064... Val Loss: 1.0325
Epoch: 40/50... Step: 4430... Loss: 1.0368... Val Loss: 1.0380
Epoch: 40/50... Step: 4440... Loss: 1.0152... Val Loss: 1.0414
Epoch: 40/50... Step: 4450... Loss: 1.0143... Val Loss: 1.0301
Epoch: 40/50... Step: 4460... Loss: 1.0339... Val Loss: 1.0319
Epoch: 40/50... Step: 4470... Loss: 0.9714... Val Loss: 1.0315
Epoch: 40/50... Step: 4480... Loss: 1.0260... Val Loss: 1.0352
Epoch: 41/50... Step: 4490... Loss: 1.0014... Val Loss: 1.0316
Epoch: 41/50... Step: 4500... Loss: 0.9925... Val Loss: 1.0367
Epoch: 41/50... Step: 4510... Loss: 0.9820... Val Loss: 1.0378
Epoch: 41/50... Step: 4520... Loss: 0.9664... Val Loss: 1.0317
Epoch: 41/50... Step: 4530... Loss: 1.0272... Val Loss: 1.0336
Epoch: 41/50... Step: 4540... Loss: 0.9772... Val Loss: 1.0325
Epoch: 41/50... Step: 4550... Loss: 0.9984... Val Loss: 1.0332
Epoch: 41/50... Step: 4560... Loss: 1.0085... Val Loss: 1.0306
Epoch: 41/50... Step: 4570... Loss: 0.9627... Val Loss: 1.0309
Epoch: 41/50... Step: 4580... Loss: 0.9722... Val Loss: 1.0308
Epoch: 41/50... Step: 4590... Loss: 0.9964... Val Loss: 1.0295
Epoch: 42/50... Step: 4600... Loss: 0.9556... Val Loss: 1.0302
Epoch: 42/50... Step: 4610... Loss: 0.9714... Val Loss: 1.0311
Epoch: 42/50... Step: 4620... Loss: 0.9520... Val Loss: 1.0317
Epoch: 42/50... Step: 4630... Loss: 1.0035... Val Loss: 1.0328
Epoch: 42/50... Step: 4640... Loss: 1.0136... Val Loss: 1.0291
Epoch: 42/50... Step: 4650... Loss: 0.9934... Val Loss: 1.0337
Epoch: 42/50... Step: 4660... Loss: 0.9574... Val Loss: 1.0340
Epoch: 42/50... Step: 4670... Loss: 0.9558... Val Loss: 1.0321
Epoch: 42/50... Step: 4680... Loss: 1.0001... Val Loss: 1.0323
Epoch: 42/50... Step: 4690... Loss: 1.0179... Val Loss: 1.0322
Epoch: 42/50... Step: 4700... Loss: 0.9858... Val Loss: 1.0292
Epoch: 43/50... Step: 4710... Loss: 0.9788... Val Loss: 1.0359
Epoch: 43/50... Step: 4720... Loss: 0.9754... Val Loss: 1.0304
Epoch: 43/50... Step: 4730... Loss: 0.9618... Val Loss: 1.0338
Epoch: 43/50... Step: 4740... Loss: 1.0036... Val Loss: 1.0330
Epoch: 43/50... Step: 4750... Loss: 0.9851... Val Loss: 1.0343
Epoch: 43/50... Step: 4760... Loss: 0.9870... Val Loss: 1.0293
Epoch: 43/50... Step: 4770... Loss: 0.9320... Val Loss: 1.0328
Epoch: 43/50... Step: 4780... Loss: 0.9319... Val Loss: 1.0361
Epoch: 43/50... Step: 4790... Loss: 0.9456... Val Loss: 1.0353
Epoch: 43/50... Step: 4800... Loss: 0.9829... Val Loss: 1.0321
Epoch: 43/50... Step: 4810... Loss: 0.9599... Val Loss: 1.0319
Epoch: 44/50... Step: 4820... Loss: 0.9798... Val Loss: 1.0342
Epoch: 44/50... Step: 4830... Loss: 0.9366... Val Loss: 1.0342
Epoch: 44/50... Step: 4840... Loss: 1.0117... Val Loss: 1.0366
Epoch: 44/50... Step: 4850... Loss: 0.9819... Val Loss: 1.0349
Epoch: 44/50... Step: 4860... Loss: 0.9842... Val Loss: 1.0343
Epoch: 44/50... Step: 4870... Loss: 0.9626... Val Loss: 1.0319
Epoch: 44/50... Step: 4880... Loss: 0.9612... Val Loss: 1.0337
Epoch: 44/50... Step: 4890... Loss: 0.9803... Val Loss: 1.0350
Epoch: 44/50... Step: 4900... Loss: 0.9677... Val Loss: 1.0327
Epoch: 44/50... Step: 4910... Loss: 0.9739... Val Loss: 1.0271
Epoch: 44/50... Step: 4920... Loss: 1.0033... Val Loss: 1.0319
Epoch: 45/50... Step: 4930... Loss: 0.9619... Val Loss: 1.0377
Epoch: 45/50... Step: 4940... Loss: 0.9724... Val Loss: 1.0323
Epoch: 45/50... Step: 4950... Loss: 0.9769... Val Loss: 1.0331
Epoch: 45/50... Step: 4960... Loss: 0.9528... Val Loss: 1.0372
Epoch: 45/50... Step: 4970... Loss: 0.9833... Val Loss: 1.0354
Epoch: 45/50... Step: 4980... Loss: 0.9840... Val Loss: 1.0345
Epoch: 45/50... Step: 4990... Loss: 1.0041... Val Loss: 1.0351
Epoch: 45/50... Step: 5000... Loss: 0.9843... Val Loss: 1.0358
Epoch: 45/50... Step: 5010... Loss: 0.9962... Val Loss: 1.0314
Epoch: 45/50... Step: 5020... Loss: 1.0016... Val Loss: 1.0326
Epoch: 45/50... Step: 5030... Loss: 0.9549... Val Loss: 1.0338
Epoch: 45/50... Step: 5040... Loss: 0.9990... Val Loss: 1.0326
Epoch: 46/50... Step: 5050... Loss: 0.9795... Val Loss: 1.0266
Epoch: 46/50... Step: 5060... Loss: 0.9733... Val Loss: 1.0320
Epoch: 46/50... Step: 5070... Loss: 0.9549... Val Loss: 1.0319
Epoch: 46/50... Step: 5080... Loss: 0.9433... Val Loss: 1.0351
Epoch: 46/50... Step: 5090... Loss: 0.9972... Val Loss: 1.0330
Epoch: 46/50... Step: 5100... Loss: 0.9518... Val Loss: 1.0348
Epoch: 46/50... Step: 5110... Loss: 0.9733... Val Loss: 1.0353
Epoch: 46/50... Step: 5120... Loss: 0.9794... Val Loss: 1.0313
Epoch: 46/50... Step: 5130... Loss: 0.9492... Val Loss: 1.0398
Epoch: 46/50... Step: 5140... Loss: 0.9452... Val Loss: 1.0376
Epoch: 46/50... Step: 5150... Loss: 0.9753... Val Loss: 1.0333
Epoch: 47/50... Step: 5160... Loss: 0.9282... Val Loss: 1.0296
Epoch: 47/50... Step: 5170... Loss: 0.9511... Val Loss: 1.0340
Epoch: 47/50... Step: 5180... Loss: 0.9309... Val Loss: 1.0369
Epoch: 47/50... Step: 5190... Loss: 0.9622... Val Loss: 1.0346
Epoch: 47/50... Step: 5200... Loss: 0.9892... Val Loss: 1.0357
Epoch: 47/50... Step: 5210... Loss: 0.9544... Val Loss: 1.0334
Epoch: 47/50... Step: 5220... Loss: 0.9281... Val Loss: 1.0327
Epoch: 47/50... Step: 5230... Loss: 0.9393... Val Loss: 1.0326
Epoch: 47/50... Step: 5240... Loss: 0.9794... Val Loss: 1.0380
Epoch: 47/50... Step: 5250... Loss: 0.9828... Val Loss: 1.0376
Epoch: 47/50... Step: 5260... Loss: 0.9558... Val Loss: 1.0339
Epoch: 48/50... Step: 5270... Loss: 0.9423... Val Loss: 1.0412
"""


"""
Hyperparameters
Here are the hyperparameters for the network.

In defining the model:

n_hidden - The number of units in the hidden layers.
n_layers - Number of hidden LSTM layers to use.
We assume that dropout probability and learning rate will be kept at the default, in this example.

And in training:

batch_size - Number of sequences running through the network in one pass.
seq_length - Number of characters in the sequence the network is trained on. Larger is better typically, the network will learn more long range dependencies. But it takes longer to train. 100 is typically a good number here.
lr - Learning rate for training
"""

model_name = 'rnn_x_epoch.net'

checkpoint = {'n_hidden': net.n_hidden,
              'n_layers': net.n_layers,
              'state_dict': net.state_dict(),
              'tokens': net.chars}

with open(model_name, 'wb') as f:
    torch.save(checkpoint, f)
    
    
def predict(net, char, h=None, top_k=None):
        ''' Given a character, predict the next character.
            Returns the predicted character and the hidden state.
        '''
        
        # tensor inputs
        x = np.array([[net.char2int[char]]])
        x = one_hot_encode(x, len(net.chars))
        inputs = torch.from_numpy(x)
        
        if(train_on_gpu):
            inputs = inputs.cuda()
        
        # detach hidden state from history
        h = tuple([each.data for each in h])
        # get the output of the model
        out, h = net(inputs, h)

        # get the character probabilities
        p = F.softmax(out, dim=1).data
        if(train_on_gpu):
            p = p.cpu() # move to cpu
        
        # get top characters
        if top_k is None:
            top_ch = np.arange(len(net.chars))
        else:
            p, top_ch = p.topk(top_k)
            top_ch = top_ch.numpy().squeeze()
        
        # select the likely next character with some element of randomness
        p = p.numpy().squeeze()
        char = np.random.choice(top_ch, p=p/p.sum())
        
        # return the encoded value of the predicted char and the hidden state
        return net.int2char[char], h
    
def sample(net, size, prime='The', top_k=None):
        
    if(train_on_gpu):
        net.cuda()
    else:
        net.cpu()
    
    net.eval() # eval mode
    
    # First off, run through the prime characters
    chars = [ch for ch in prime]
    h = net.init_hidden(1)
    for ch in prime:
        char, h = predict(net, ch, h, top_k=top_k)

    chars.append(char)
    
    # Now pass in the previous character and get a new one
    for ii in range(size):
        char, h = predict(net, chars[-1], h, top_k=top_k)
        chars.append(char)

    return ''.join(chars)

print(sample(net, 1000, prime='Krsna', top_k=5))
"""
Krsna who is not
due to his servant and therefore accepts the peace of the special fruitive
results with a tree is the supreme abode.

The Lord at once the persons in the mode of ignorance, the different
material attempts who are diverted by a demigod is always always always transcendental. This
is the cause, because it is the spiritual sky. As sons a compansion is
the subject matter of material activities. A person is there in the senses of
the sun, which is considered to as the right of the body.

TEXT 16

ﬁaﬁﬁwﬁﬁaﬁmmww
WWHWHWQHII

param caivc‘tdmyc‘mi sc‘tm
aham avyavatc‘ttmc‘t

tad viddhi samyamya
sadvair s’raddhayc‘wc‘tmanam

SYNONYMS

ye—who; aham—I; sukhc‘mtm—to take transcendental loving service;
yat—that which; pas’yati—sees; samyamya—sacrifice; sahatram—standing;
paramc‘t—bhaktc‘th—being freed from; as’a—buddhih—intelligence; ye—those who;
param—transcendental; pas’yati—sees; samatc‘tm—commins; ksetram—the different
way; papam—the material world; mahi—tate—the desired senses;
prakr
"""
with open('rnn_x_epoch.net', 'rb') as f:
    checkpoint = torch.load(f)
    
loaded = CharRNN(checkpoint['tokens'], n_hidden=checkpoint['n_hidden'], n_layers=checkpoint['n_layers'])
loaded.load_state_dict(checkpoint['state_dict'])
print(sample(loaded, 2000, top_k=5, prime="Krsna told Arjuna"))
"""
Krsna told Arjuna,
that at the end of the Brahmaloka, although a devotee also constitute the
pure devotional service in the service of the Lord and in Krsna consciousness are
attached to the Supreme Lord. Then there are different costimes, then
there are superior things, and the mind is fully conscious of
all the activities of the meant to actually accept the stage of devotional
service. All the persons are always free from the activities of the descriptions of
the Vedic injunctions they want to be steady in self-realization. Also also
sees all kinds of bodies and any mercy there are always watted and the
source of trees, and which indicates the standard of material nature, the
mind is therefore accepted by such devotional serVice. Then there are
supreme divine practices of the material world and the sun is chanting here to
the sees of the Supreme Lord, but the Supreme Person, as a person
mundane scholas, who are in the statumen should thinking of the Lord
constantly engaged in the material world. It is not possible for the Supreme
Lord and the sons of Dhrtarastra, beyond this part, the moon and the mind as a
spiritual sky.

TEXT 36
WWEIWWI
mammmmaﬁrmzueu

kali—bhaktarc‘tthc‘t bhc‘wim
adhyc‘ttmahc‘t matato manye
yathc‘th purc‘ma—vidhai

sahasra—bhﬁti samatc‘th

SYNONYMS

yuktah—engaged in the supply; anta—c‘ttmakam—the sons of Dhrtarastra;
su—his’cas’yc‘tm—of all state; as’vattham—and acting.

TRANSLATION

Arjuna said:



yogi mam asthe tad bhavati
karma—krsna-bhﬁtc‘tthc‘i—param
parasya s’c‘iktam s’c‘ts’vati s’c‘mtim pratis’yati

“The perfect yogi is not perfectly enlygred in the devotional service
of the Lord, and he is called the same platform, he can come to the stage of
particular propers endeavor. The devotee are those who are eternal. It is
said. Ivara—para—devc‘m) in the strength that of this pracess is not the spiritual
master and who are described in the Vedas into the Vedas and all these
transcendental loving service to the Lord.

TEXT 16

WWWWI
WWWWWWIIQQII

mat—param anas
"""

print(sample(loaded, 2000, top_k=5, prime="the supreme personality of godhead"))
"""
the supreme personality of godhead in
this way. This is a state, the devotee in the supreme species.

TEXT 27

aﬁiﬁﬁwﬁﬁﬁmwﬁaaﬁml
ammamaaawzﬁuqu

s’ubham kseta—ksame maya
yajﬁa—parandirc‘tm s’ri—ksarih
samah bhavati ksetram

yasya prajc‘ttatc‘tnate tathc‘ttmam

SYNONYMS

s’ri—bhagavc‘m uvc‘tca—the Supreme Personality of Godhead;
sattam—the senses; sukham—that illusion; tad—that—and then the mission;
asma—of the body; s’as’i—accepting; parameta—surrender;
s’retim—proguced.

TRANSLATION

One who is freed from the and spiritual world have stated hy the supreme
distinctions of the Supreme Personality of Godhead by material constatusing.

PURPORT

Arjuna was adrows to be the Supreme Lord.

There are three demigods whereas impersonalists are to approach the
Supreme Personality who is nectiral because if a devotee sees thir statement
is the creator in all modes and the mind.

If one is actually situated in Krsna consciousness in Krsna
consciousness can attain the highest perfectional stage. Such immediate terribs
are complete specific detieed by must be steady in a first attachment with
Krsna, which is the spiritual sky there is no cause and soll factually
and having a demonic persons according to the modes of material
nature. This consciousness will never be seen and what has born and is
always engaged in material nature, and the devotees are to be considered the
same to the supreme spiritual master, in the spiritual world the soul is
activities in such secreting hir detreiss enter.

Although He is above him in His benefic are contaminated by His mind, and
therefore He is transcendental. Actually all of them are devoted, they are the
civilized comminination in the Bhagavad-gitc‘t, and there is no sees. The devotee
does not teliving the sins on the process, whatever he wants to back. The demigods there
is no leader, but if he is thinking of some different transcendental services as
more immediately and the partial individual living entity is actually
also a devotee of this material manifestation. Thar is als
"""
print(sample(loaded, 2000, top_k=5, prime="the three modes of material nature"))
"""
the three modes of material nature and
the superior to be the progressive are allowed to how bat the person, the individual
soul is not impersonal. If one is not absorbed in the practice of yoga
progressing on the spiritual world, whereas the lowest and cannot self
the sound, a devotee of the Lord, is to carriel his particular three
wishom. Those who are serious about something is situated in a
spiritual sky is the cause of all causes; the Supersoul are so perfect that the
Supreme Lord wants to be fixed in Krsna consciousness; therefore the
many pooper man could not become action, the material energy and he is
not the process of devotional serVice that the Supreme Soul is spiritual
existences for the spiritual master, because He is the origin of all living
entities with all statements in the same alsa. The Lord is eternally an
explained sage for sense gratification. And in the Brahma-samhitc‘t as
the Supreme Personality of Godhead is the Supreme Personality of
Godhead and the supreme enjoyer. It is not the sublime of the brahmajyoti,
which are disciple their material attachment, then the seed are to
be tooge any person in Krsna consciousness as so there is to speak of
all the soul, but they are so more accepted in all position. Therefore, in the
Brabma-san'ilu'ta (5.17—3) it is clearly stated that
the Lord says, tamasya s’c‘tsya sa bhc‘wah dhiham. Therefore, the Lord
is a man in Krsna consciousness, one has to attain the material body.
There is no cause, which is called a mahc'ttmc‘t. Therefore the demigods can
understands the state of the Supreme Lord within the conditioned life on
the transcendental loving service of the Lord. As the soul is called a
past man, and there is no caure. Thus his material not tollight,
he does not belong to the search of a substance and that only he wants to
become an instruction of the brc‘thmahas, and they are all annihilated. One
should take their perfections which are attached to
Krsna in the mode of passion, but the mode and
servant of the Supreme Lord, the person 
"""

