import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.autograd import Variable

class CNN_Kim2014(nn.Module):

  def __init__(self, embed_num, label_num, embeddings_dim, embeddings_mode, initial_embeddings, kernel_width, feature_num):
    
    # cannot assign module before Module.__init__() call
    super(CNN_Kim2014, self).__init__()

    self.embed_num    = embed_num  # vocab size
    self.label_num    = label_num
    self.embed_dim    = 300        # mengikuti dimensi embedding Mikolov 2013
    self.embed_mode   = embeddings_mode
    self.channel_in   = 1
    self.feature_num  = feature_num
    self.kernel_width = kernel_width
    self.dropout_rate = 0.5
    self.norm_limit   = 3

    # print('kw',self.kernel_width)
    # print('fn',self.feature_num)
    # exit()

    # print(self.embed_mode)
    # exit()

    assert (len(self.feature_num) == len(self.kernel_width))

    self.kernel_num = len(self.kernel_width) 
    # sesuai dengan jumlah filter (single / multiple)

    self.embeddings = nn.Embedding(self.embed_num, self.embed_dim, padding_idx=1)

    # proses embedding, input nya jml vocab, sama dimensi vector, outpunya class embedding dengan parameter input yang diberikan
    # initial_embeddings adalah output dari proses pembobotan vocab dari model word embedding sebelumnya     

    self.embeddings.weight.data.copy_(torch.from_numpy(initial_embeddings))

    # proses mengcopy weight / vector vocab dari model word embedding

    if self.embed_mode == 'static':

        self.embeddings.weight.requires_grad = False

        # kalo static pake stocastic gradient

    # a = 5

    # b = [num*a for num in range(1,10)]

    # print(b)

    # for i in range(self.kernel_num):
    #     print(self.channel_in, self.feature_num[i], (self.embed_dim*self.kernel_width[i]), self.embed_dim)

    # conv = nn.Conv1d(self.channel_in, self.feature_num[0],self.embed_dim*self.kernel_width[0], stride=self.embed_dim)
    # print(conv)

    convo = [nn.Conv1d(self.channel_in, self.feature_num[i],self.embed_dim*self.kernel_width[i], stride=self.embed_dim) for i in range(self.kernel_num)]

    # print(convo[0])

    # print(convo[1])

    self.convs = nn.ModuleList(convo)

    self.linear = nn.Linear(sum(self.feature_num), self.label_num)

    # print(self.linear)

  def forward(self, input):


    batch_width = input.size()[1]

    # for x in range(0,len(input)):
    #     print(input[x])
    #     break

      

    # print(batch_width)

    # print('Asli', Variable(input).size())

    x = self.embeddings(input).view(-1, 1, self.embed_dim*batch_width)

    # print('Embed', Variable(x).size())

    # # print('Embed 0', Variable(x[0]).size())

    # # print(x[0])

    # z = self.convs[0](x)

    # print('Convo', Variable(z).size())

    # # print('Convo 0', Variable(z[0]).size())

    # # print(z[0])

    # # y = self.convs[1](x)

    # # print('Convo', Variable(y).size())

    # # print('Convo 1', Variable(y[0]).size())

    # # print(y[0])
    # # print(z)

    # # print(z[0][0])

    # b = F.relu(z)

    # print('Relu',Variable(b).size())

    # # print(batch_width,self.kernel_width[0],batch_width - self.kernel_width[0] + 1)

    # j = batch_width - self.kernel_width[0] + 1

    # t = [F.max_pool1d(b,j).view(-1,self.feature_num[0])]

    # print('max_pool1d', Variable(t[0]).size())

    # # print(t)
    # n = torch.cat(t, 1)

    # # print(p)

    # lp = F.dropout(n, p=self.dropout_rate, training=self.training)

    # ko = self.linear(lp)

    # print(ko)
    # # print(lp)

    # exit()

    conv_results = [ #proses konvolusi, out channel menghasilkan feature maps, pake rumus s-reg+1, terus di relu terus di max pool jadi 1 syntax

       F.max_pool1d(F.relu(self.convs[i](x)), batch_width - self.kernel_width[i] + 1).view(-1, self.feature_num[i])

       for i in range(len(self.feature_num))
    ]

    x = torch.cat(conv_results, 1)
    x = F.dropout(x, p=self.dropout_rate, training=self.training)
    x = self.linear(x)

    return x