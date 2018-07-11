import argparse
import model
import prepare_data
import torch
import numpy as np
import read_embed
import inspect
import math
import torch.nn as nn
import torch.nn.functional as F
import torchtext.data
from collections import defaultdict
from math import ceil
import statistics
import time
import datetime
import string
import random
import sys
from torch.autograd import Variable
import pandas as pd

parser  = argparse.ArgumentParser(description='Hopefully somewhat generalized classifier for Sentiment Analysis')

parser.add_argument('-embeddings_dim', type=int, default=300, help='the default size of embeddings. '
                                                                   'When mikolov2013 is used, must use a size of 300, '
                                                                   'when godin2015 is used, must use a size of 400, '
                                                                   'and when grave2018 is used, must use 300 ')
parser.add_argument('-epoch_num', type=int, default=10, help='number of epochs')
parser.add_argument('-fold_num', type=int, default=10, help='number of folding')
parser.add_argument('-embeddings_mode', type=str, default='non-static', help='random, static,non-static, or multichannel')
parser.add_argument('-model_type', type=str, default='kim2014', help='kim2014')
parser.add_argument('-data', type=str, default='indo', help='datanya')
parser.add_argument('-embed', type=str, default='w2v', help='tipe word embedding')
parser.add_argument('-batch_size', type=int, default=50, help='size of the minibatches')
parser.add_argument('-cuda', type=bool, default=False, help='whether to use GPU')
parser.add_argument('-subt', type=bool, default='bc', help='Subtask BC default nya')


args  = parser.parse_args()

def cross_validate(fold, data, embeddings, args, key_vector, kernel_width, feature_num, batch_size,embeddings_mode, subtask):
  
  
  actual_counts     = defaultdict(int)
  predicted_counts  = defaultdict(int)
  match_counts      = defaultdict(int)

  split_width = int(ceil(len(data.examples)/fold))

  # for x in data.examples:
  #   print(x.text,x.label)
  #   break
  # exit()

  for i in range(fold):

    # print('###=============FOLD [{}]=============###'.format(i + 1))
    
    train_examples  = data.examples[:]

    # print(len(data.examples))
    # exit()

    del train_examples[i*split_width:min(len(data.examples), (i+1)*split_width)]

  
    test_examples   = data.examples[i*split_width:min(len(data.examples), (i+1)*split_width)]
    
    # print('###---------------Counts--------------###')

    total_len = len(data.examples)
    train_len = len(train_examples)
    test_len  = len(test_examples)

    train_counts = defaultdict(int)
    test_counts  = defaultdict(int)

    for example in train_examples:

      train_counts[example.label] += 1
    
    for example in test_examples:

      test_counts[example.label] += 1


    # print('\tTotal Number of Examples : {}\n'.format(total_len))
    # print('\tNumber of Train Examples : {}'.format(train_len))

    strg_pos  = train_counts['strong positive']
    pos       = train_counts['positive']
    neu       = train_counts['neutral']
    neg       = train_counts['negative']
    strg_neg  = train_counts['strong negative']

    # print('\t\tTrain-Strong Positive  = {} ({}%)'.format(strg_pos, 100*strg_pos/train_len))
    # print('\t\tTrain-Positive         = {} ({}%)'.format(pos, 100*pos/train_len))
    # print('\t\tTrain-Neutral          = {} ({}%)'.format(neu, 100*neu/train_len))
    # print('\t\tTrain-Negative         = {} ({}%)'.format(neg, 100*neg/train_len))
    # print('\t\tTrain-Strong Negative  = {} ({}%)\n'.format(strg_neg, 100*strg_neg/train_len))
    
    # print('\tNumber of Test Examples : {}'.format(test_len))


    strg_pos  = test_counts['strong positive']
    pos       = test_counts['positive']
    neu       = test_counts['neutral']
    neg       = test_counts['negative']
    strg_neg  = test_counts['strong negative']

    # print('\t\tTest-Strong Positive  = {} ({}%)'.format(strg_pos, 100*strg_pos/test_len))
    # print('\t\tTest-Positive         = {} ({}%)'.format(pos, 100*pos/test_len))
    # print('\t\tTest-Neutral          = {} ({}%)'.format(neu, 100*neu/test_len))
    # print('\t\tTest-Negative         = {} ({}%)'.format(neg, 100*neg/test_len))
    # print('\t\tTest-Strong Negative  = {} ({}%)\n'.format(strg_neg, 100*strg_neg/test_len))
    # print('###-----------------------------------###')

    fields = data.fields

    train_set = torchtext.data.Dataset(examples=train_examples, fields=fields)
    test_set  = torchtext.data.Dataset(examples=test_examples, fields=fields)

    all_set = torchtext.data.Dataset(examples=data.examples, fields=fields)    

    

    # exit()

    # for x in fields:
    #   print(x)


    text_field  = None
    label_field = None

    for field_name, field_object in fields:

      if field_name == 'text':
        text_field = field_object

      elif field_name == 'label':
        label_field = field_object
        # print(field_object)
    
    # print(len(train_set),len(test_set),len(all_set))

    text_field.build_vocab(train_set)
    label_field.build_vocab(train_set)

    # print(len(text_field.vocab.stoi))
    # exit()

    data.vocab_to_idx = dict(text_field.vocab.stoi)
    data.idx_to_vocab = {v: k for k, v in data.vocab_to_idx.items()}


    data.label_to_idx = dict(label_field.vocab.stoi)
    data.idx_to_label = {v: k for k, v in data.label_to_idx.items()}

    embed_num = len(text_field.vocab)
    label_num = len(label_field.vocab)

    print(data.idx_to_label)


    # print(label_num)
    # print(embed_num)
    # exit()

    known_word, unknown_word, emb_init_values = data.create_fold_embeddings(embeddings, args, key_vector)
    emb_init_values = np.array(emb_init_values)

    train_iter, test_iter = torchtext.data.Iterator.splits((train_set, test_set),batch_sizes=(batch_size, len(test_set)),device=-1,repeat=False,shuffle=False)
    #split train set menjadi batch_size (50) dan test_set menjadi batch sejumlah data testnya yaitu 1160

    # print(len(train_set),len(test_set))


    # for x in train_set:
    #   print(x.text)
    #   break

    # for x in test_set:
    #   print(x.text)
    #   break

    # f = 1;
    # for x in train_iter:
    #   f += 1

    # print(len(test_set))

    # # train_iter.sort_key = lambda x: len(x.text)

    # for x in train_iter:
    #   a = x.text
    #   # print(Variable(x.text).size())
    #   break

    # print(a)

    
   # print('Known Word :', known_word)
	 # print('Unknown Word :', unknown_word)

	 # exit()

    train_bulk_dataset = train_set,
    train_bulk__size   = len(train_set),

    train_bulk_iter    = torchtext.data.Iterator.splits(datasets=train_bulk_dataset, batch_sizes=train_bulk__size,device=-1, repeat=False)[0]

    train_bulk_iter.sort_key = lambda x: len(x.text)

    # for batch in train_bulk_iter:
    #   print(Variable(batch.text).size())

    # for batch in train_iter:
    #   print(batch)

    # exit()
    
    kim2014 = model.CNN_Kim2014(embed_num, label_num - 1,args.embeddings_dim, embeddings_mode,emb_init_values,
    										kernel_width, feature_num)

    if args.cuda:
        kim2014.cuda()
    
    # print(data.idx_to_vocab[1])
    # print(data.label_to_idx)

    trained_model = train(kim2014, train_iter, test_iter, data.label_to_idx, data.idx_to_label, train_bulk_iter,i,subtask)
    # exit()


  
  return known_word, unknown_word, kim2014
  

    # exit()


def train(model, train_iter, test_iter, label_to_idx, idx_to_label, train_bulk_iter,fold,subtask):
  
  parameters = filter(lambda p: p.requires_grad, model.parameters())

  optimizer = torch.optim.Adadelta(parameters)

  if args.cuda:
    model.cuda()
  
  model.train()


  for epoch in range(1, args.epoch_num+1):
    
    # print("###__________FOLD/EPOCH [{}/{}]__________###".format(fold+1,epoch))
    steps = 0
    corrects_sum  = 0

    go = time.time()
    
    for batch in train_iter:
      text_numerical, target = batch.text, batch.label

      if args.cuda:
        text_numerical, target = text_numerical.cuda(), target.cuda()

      text_numerical.data.t_()
      target.data.sub_(1)

      optimizer.zero_grad()

      

      forward = model(text_numerical)

      
      loss = F.cross_entropy(forward, target)
      loss.backward()
      optimizer.step()
      steps += 1

      corrects = (torch.max(forward, 1)[1].view(target.size()).data == target.data).sum()
      
      accuracy = 100.0 * corrects / batch.batch_size

      # print('\rFold[{}] - Epoch[{}] - Batch[{}] - loss: {:.6f}  acc: {:.4f}%({}/{})\n'.format(fold,epoch,steps, loss.data[0],accuracy, corrects,batch.batch_size))
    
    # print('\n###____Performance on Train Data____###')
    # exit()

    actual, predicted, acc = evaluate(model, train_bulk_iter,'training',epoch,fold)

    epoch_actual_counts, epoch_predicted_counts, epoch_match_counts, actual, predicted = calculate_fold_counts(actual, predicted, label_to_idx, idx_to_label,'training')
    
    display_measures(acc, epoch_actual_counts, epoch_predicted_counts, epoch_match_counts,actual,predicted,idx_to_label,'training',epoch,fold,go,subtask)
    
    # print('\n###____Performance on Test Data____###')
    
    actual, predicted, acc = evaluate(model, test_iter,'testing',epoch,fold)
    
    epoch_actual_counts, epoch_predicted_counts, epoch_match_counts, actual, predicted = calculate_fold_counts(actual, predicted, label_to_idx, idx_to_label,'testing')
    
    display_measures(acc, epoch_actual_counts, epoch_predicted_counts, epoch_match_counts,actual,predicted,idx_to_label,'testing',epoch,fold,go,subtask)

    # exit()
  
  return model
    
def evaluate(model, data_iter,type,epoch,fold):

  model.eval()
  corrects, avg_loss = 0, 0

  
  data_iter.sort_key = lambda x: len(x.text)

  for batch in data_iter:
    
    text_numerical, target = batch.text, batch.label

    if args.cuda:
        text_numerical, target = text_numerical.cuda(), target.cuda()

    text_numerical.data.t_()
    target.data.sub_(1)

    forward = model(text_numerical)
    loss = F.cross_entropy(forward, target, size_average=False)

    avg_loss += loss.data[0].item()
    corrects += (torch.max(forward, 1)[1].view(target.size()).data == target.data).sum()

  size = len(data_iter.dataset)
  avg_loss = avg_loss/size

  accuracy = 100.0 * corrects/size

  cor = corrects.item()

  acc = 100 * (cor/size)

  if type == 'testing':


    track_accuracy.append([fold+1,epoch,acc])

  return target.data, torch.max(forward, 1)[1].view(target.size()).data, acc


def calculate_fold_counts(actual, predicted, label_to_idx, idx_to_label, type):

  assert len(actual)  ==  len(predicted)

  # print(label_to_idx)
  # exit()

  # res_train = []
  # res_test = []

  # res_train.clear()
  # res_test.clear()

  fold_actual_counts    = defaultdict(int)
  fold_predicted_counts = defaultdict(int)
  fold_match_counts     = defaultdict(int)

  # mae_calculate_label   = defaultdict(int)
  # std_calculate_label   = defaultdict(float)

  

  # exit()



  if args.cuda:

    for i in range(len(actual)):

        idx   = actual[i] + 1
        label = idx_to_label[idx.item()]
        fold_actual_counts[label] += 1

        # diff_label = abs(predicted[i]-actual[i])
        # diff_mean  = (idx-mean)**2

        # mae_calculate_label[label] += diff_label
        # std_calculate_label[label] += diff_mean

        # if type == 'training':
        # 	res_train.append(idx.item())
        # else:
        # 	res_test.append(idx.item())

        if actual[i] == predicted[i]:
          fold_match_counts[label] += 1
      
    for i in range(len(predicted)):

        idx = predicted[i] + 1
        label = idx_to_label[idx.item()]
        fold_predicted_counts[label] += 1

  else:

    for i in range(len(actual)):

        idx   = actual[i] + 1
        label = idx_to_label[idx]
        fold_actual_counts[label] += 1

        if actual[i] == predicted[i]:
          fold_match_counts[label] += 1
      
    for i in range(len(predicted)):

        idx = predicted[i] + 1
        label = idx_to_label[idx]
        fold_predicted_counts[label] += 1

	# if type == 'training':
	# 	st1 = statistics.stdev(res_train)
	# else:
	# 	st2 = statistics.stdev(res_test)

  return fold_actual_counts, fold_predicted_counts, fold_match_counts, actual, predicted

def display_measures(acc, actual_counts, predicted_counts, match_counts,actual,predicted,idx_to_label,type,epoch,fold,go,subtask):
  
  precisions = defaultdict(float)
  recalls    = defaultdict(float)
  f_measures = defaultdict(float)

  

  # exit()

  # standard_mae = 0
  # test_size = sum(actual_counts.values())

  # ac = []
  # mc = []

  mae_temp = 0
  label_count = 0

  rc_temp2 = 0
  pr_temp2 = 0
  fm_temp2 = 0

  predicted_temp = []
  predicted_temp.clear()

  for i in range(len(predicted)):
      
    pred = predicted[i] + 1
    predicted_temp.append(pred.item())

  # tot = 0
  # for label in actual_counts.keys():
  # 	tot += actual_counts[label]

  kld_temp = []
  kld_temp.clear()

  for label in actual_counts.keys():

    # macro_mae_avg = mae_calculate_label[label] / actual_counts[label] if actual_counts[label] > 0 else 0
    # standard_mae  += mae_calculate_label[label]
    
    # std_avg = std_calculate_label[label] / (actual_counts[label]-1) if (actual_counts[label]-1) > 0 else 0

    # kld_actual    = actual_counts[label]/test_size
    # kld_predicted = predicted_counts[label]/test_size
    # diff = kld_actual/kld_predicted if kld_predicted > 0 else 0
    # kld_calculate = kld_actual * math.log(diff) if diff > 0 else 0

    precision = match_counts[label] / predicted_counts[label] if predicted_counts[label] > 0 else 0
    recall    = match_counts[label] / actual_counts[label] if actual_counts[label] > 0 else 0
    f_measure = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

    rc_temp2 += 100 * recall
    pr_temp2 += 100 * precision
    fm_temp2 += 100 * f_measure

    precisions[label] = 100 * precision
    recalls[label]    = 100 * recall
    f_measures[label] = 100 * f_measure

    jml = 0
    jml_pred = 0
    avg_mae = 0

    for i in range(len(actual)):
      
      act = actual[i] + 1
      pred = predicted[i] + 1
      label_act = idx_to_label[act.item()]
      label_pred = idx_to_label[pred.item()]
      
      if label_act == label:
        
        delta = abs(act.item()-pred.item())
        avg_mae += delta
        jml += 1

      if label_pred == label:

        jml_pred += 1

    p = jml/len(actual)
    px = jml_pred/len(actual)

    kldx = [p,px,label]
    kld_temp.append(kldx)
    
    mae_label = avg_mae/jml
    mae_temp += mae_label

    if type == 'testing':

      track_recall.append([fold+1,epoch,label,100 * recall,'recall'])
      track_fmeasure.append([fold+1,epoch,label,100 * f_measure,'f_measure'])
      track_precission.append([fold+1,epoch,label,100 * precision,'precision'])

    # print('Lebel :',label,' jml data aktual : ',actual_counts[label])
    # print('Lebel :',label,' jml data prediksi : ',predicted_counts[label])

    # a = actual_counts[label]/tot
    # b = predicted_counts[label]/tot
    # # print(b)
    # if b == 0:
    # 	c = 0
    # else:
    # 	c = math.log(a/b)
    # # print(c)
    # d = a*c

    # print(d)
    # print("On class {}:".format(label))
    # print("\tPrecision = {}%".format(100 * precision))
    # print("\tRecall    = {}%".format(100 * recall))
    # print("\tF-Measure = {}%".format(100 * f_measure))
    # print("\tMacro-MAE  = {}".format(mae_label))

    label_count += 1
  
  KLD = 0

  if subtask == 'A':

    p1 = float(kld_temp[0][0])
    p1x = float(kld_temp[0][1])
    p2 = float(kld_temp[1][0])
    p2x = float(kld_temp[1][1])

    logp1 = math.log(p1/p1x)
    logp2 = math.log(p2/p2x)

    KLD = (p1*logp1) + (p2*logp2)

  else:

    KLD = 0


  mae = mae_temp/label_count
  stdev = statistics.stdev(predicted_temp)

  if type == 'testing':

    track_kld.append([fold+1,epoch,KLD])
    track_mae.append([fold+1,epoch,mae])
    track_stdev.append([fold+1,epoch,stdev])

  mae = str(round(mae_temp/label_count, 3))
  rc = str(round(rc_temp2/label_count, 2))
  pr = str(round(pr_temp2/label_count, 2))
  fm = str(round(fm_temp2/(label_count-1), 2))
  acc = str(round(acc, 2))
  stdev = str(round(stdev, 3))
  kld = str(round(KLD,7))

  if type == 'testing' and epoch == args.epoch_num:


    x = (Variable(actual).data).cpu().sub(-1).numpy()
    y = (Variable(predicted).data).cpu().sub(-1).numpy()
    
    y_actu = pd.Series(x, name='Actual')
    y_pred = pd.Series(y, name='Predicted')
    df_confusion = pd.crosstab(y_actu, y_pred)
    print(df_confusion)
    confusion.append(df_confusion.values)

  if type == 'testing':

    finish = time.time()
    dur = finish - go
    print("SC : {} FO : {} EP : {} \t ACC : {} \t RC : {} \t PR : {} \t FM : {} \t MAE : {} \t STDEV : {} \t KLD : {} \t TIME : {} \t".
      format(index_scenario,fold+1,epoch,acc,rc,pr,fm,mae,stdev,kld,dur))



def play_scenario(width, feature, batch_size, subtask):

  
  print(' ')
  print('====================================')
  print('--==$x$==-- CEK 2 SCENARIO {} --==$x$==--'.format(index_scenario))
  print('====================================')
  print(' ')

  f.write(' \n')
  f.write('====================================\n')
  f.write('--==$x$==-- SCENARIO {} --==$x$==--\n'.format(index_scenario))
  f.write('====================================\n')
  f.write(' \n')

  
  args.model_type  = 'kim2014'
  # args.embed_mode  = 'non-static'
  args.embeddings_source = 'none'

  print(' ')
  print('--- INITIAL PROPERTIES ---')
  print(' ')

  print('Scenario Code       : {}'.format(scenario_code))
  print('Subtask             : {}'.format(subtask))
  print(' ')
  print('Model type       : {}'.format(args.model_type))
  print('Embedding Mode   : {}'.format(args.embeddings_mode))
  print('Number of Epochs : {}'.format(args.epoch_num))
  print('Batch Size       : {}'.format(batch_size))
  print('Folding Size     : {}'.format(args.fold_num))
  print('Run Using GPU    : {}'.format(args.cuda))

  print(' ')
  print('--- MODEL INITIAL PROPERTIES ---')
  print(' ')

  print('kernel_width    : {}'.format(width))
  print('feature_num     : {}'.format(feature))

  print(' ')
  print('--- EMBEDDING PROPERTIES ---')
  print(' ')

  print('embed_cat      : {}'.format(args.embed))

  f.write(' \n')
  f.write('--- INITIAL PROPERTIES ---\n')
  f.write(' \n')

  f.write('Scenario Code       : {}\n'.format(scenario_code))
  f.write('Subtask             : {}'.format(subtask))
  f.write(' \n')
  f.write('Model type       : {}\n'.format(args.model_type))
  f.write('Embedding Mode   : {}\n'.format(args.embeddings_mode))
  f.write('Number of Epochs : {}\n'.format(args.epoch_num))
  f.write('Batch Size       : {}\n'.format(batch_size))
  f.write('Folding Size     : {}\n'.format(args.fold_num))
  f.write('Run Using GPU    : {}\n'.format(args.cuda))

  f.write(' \n')
  f.write('--- MODEL INITIAL PROPERTIES ---\n')
  f.write(' \n')

  f.write('kernel_width    : {}\n'.format(width))
  f.write('feature_num     : {}\n'.format(feature))


  in_data = prepare_data.prepare_data()
  in_data.read_dataset(args.data,subtask)

  embeddings = read_embed.Embeddings()
  key_vector = embeddings.read_model(args.data, args.embed)
  args.embeddings_dim = embeddings.embed_dim

  track_recall.clear()
  track_precission.clear()
  track_fmeasure.clear()

  track_accuracy.clear()
  track_mae.clear()
  track_stdev.clear()
  track_kld.clear()

  confusion.clear()

  print(' ')
  print('--- EXECUTION TIME ---')
  print(' ')

  start = time.time()
  
  known_word, unknown_word, kim2014 = cross_validate(args.fold_num, in_data, embeddings, args, key_vector, width, feature,batch_size,args.embeddings_mode,subtask)
  
  end = time.time()
  
  print(' ')
  print('--- MODEL FINAL PROPERTIES () ---')
  print(' ')
  
  print('embed_num :', kim2014.embed_num)
  print('label_num :', kim2014.label_num)
  print('embed_dim :', kim2014.embed_dim)
  print('embed_mode :', kim2014.embed_mode)
  print('channel_in :', kim2014.channel_in)
  print('feature_num :', kim2014.feature_num)
  print('kernel_width :', kim2014.kernel_width)
  print('dropout_rate :', kim2014.dropout_rate)
  print('norm_limit :', kim2014.norm_limit)

  print(' ')
  print('--- WORDEMBED PROPERTIES (kim2014s) ---')
  print(' ')

  print('Known Word :', known_word)
  print('Unknown Word :', unknown_word)

  f.write(' \n')
  f.write('--- MODEL FINAL PROPERTIES (kim2014s) ---\n')
  f.write(' \n')
  
  f.write('embed_num : {}\n'.format(kim2014.embed_num))
  f.write('label_num : {}\n'.format(kim2014.label_num))
  f.write('embed_dim : {}\n'.format(kim2014.embed_dim))
  f.write('embed_mode : {}\n'.format(kim2014.embed_mode))
  f.write('channel_in : {}\n'.format(kim2014.channel_in))
  f.write('feature_num : {}\n'.format(kim2014.feature_num))
  f.write('kernel_width : {}\n'.format(kim2014.kernel_width))
  f.write('dropout_rate : {}\n'.format(kim2014.dropout_rate))
  f.write('norm_limit : {}\n'.format(kim2014.norm_limit))

  f.write(' \n')
  f.write('--- WORDEMBED PROPERTIES (kim2014s) ---\n')
  f.write(' \n')

  f.write('Known Word : {}\n'.format(known_word))
  f.write('Unknown Word : {}\n'.format(unknown_word))
  f.write('embed_cat      : {}\n'.format(args.embed))

  epoch = args.epoch_num
  fold = args.fold_num
  
  label_distinct = []
  epoch_num = args.epoch_num

  for num in range(0,len(track_recall)):
    label = track_recall[num][2] 
    label_distinct.append(label)

  label_distinct = list(set(label_distinct))

  print(' ')
  print("Label Detected    : {}".format(label_distinct))
  print(' ')

  f.write(' \n')
  f.write("Label Detected    : {}\n".format(label_distinct))
  f.write(' \n')

  avg_temp = []
  sorted_perfomance = []
  
  print(' ')
  print('--- RECALL, PRECISION, F-MEASURE RESULT ---')
  print(' ')

  rc_temp = []
  pr_temp = []
  fm_temp = []

  for x in range(0,len(label_distinct)):
    
    label = label_distinct[x]

    rc_temp.clear()
    pr_temp.clear()
    fm_temp.clear()
    
    print(' ')
    print('---- label : {} ----'.format(label))
    print(' ')

    for z in range(1,epoch_num+1):

      print(' ')
      print('---- epoch : {} ----'.format(z))
      print(' ')

      avg_temp.clear()      

      for y in range(0,len(track_recall)):

        if label == track_recall[y][2] and z == track_recall[y][1]: 

          print(track_recall[y])
          sorted_perfomance.append(track_recall[y])
          avg_temp.append(track_recall[y][3])

      print(' ')
      print('AVG Recall ',sum(avg_temp) / float(len(avg_temp)))
      avg_perfomance.append([index_scenario,z,label,sum(avg_temp) / float(len(avg_temp)),'AVG Recall'])
      rc_temp.append([sum(avg_temp) / float(len(avg_temp))])
      print(' ')

      avg_temp.clear()

      for y in range(0,len(track_precission)):

        if label == track_precission[y][2] and z == track_precission[y][1]: 

          print(track_precission[y])
          sorted_perfomance.append(track_precission[y])
          avg_temp.append(track_precission[y][3])

      print(' ')
      print('AVG Precision ',sum(avg_temp) / float(len(avg_temp)))
      avg_perfomance.append([index_scenario,z,label,sum(avg_temp) / float(len(avg_temp)),'AVG Precision'])
      pr_temp.append([sum(avg_temp) / float(len(avg_temp))])
      print(' ')

      avg_temp.clear()

      for y in range(0,len(track_fmeasure)):

        if label == track_fmeasure[y][2] and z == track_fmeasure[y][1]: 

          print(track_fmeasure[y])
          sorted_perfomance.append(track_fmeasure[y])
          avg_temp.append(track_fmeasure[y][3])

      print(' ')
      print('AVG F-Measure ',sum(avg_temp) / float(len(avg_temp)))
      avg_perfomance.append([index_scenario,z,label,sum(avg_temp) / float(len(avg_temp)),'AVG F-Measure'])
      fm_temp.append([sum(avg_temp) / float(len(avg_temp))])
      print(' ')

    report_perfomance.append([index_scenario,label,'Recall',rc_temp])
    report_perfomance.append([index_scenario,label,'Precision',pr_temp])
    report_perfomance.append([index_scenario,label,'F-Measure',fm_temp])
    

  print(' ')
  print('--- ACCURACY, MAE, STDEV, KLD STATISTIC ---')
  print(' ')

  acc_temp = []
  acc_temp.clear()

  mae_all_temp = []
  mae_all_temp.clear()

  stdev_temp = []
  stdev_temp.clear()

  kld_temp = []
  kld_temp.clear()

  for x in range(1,epoch_num+1):

    print(' ')
    print('---- epoch : {} ----'.format(x))
    print(' ')
    
    avg_temp.clear()

    for y in range (0, len(track_accuracy)):

      if x == track_accuracy[y][1]:

        print(track_accuracy[y])
        sorted_perfomance.append([track_accuracy[y][0],track_accuracy[y][1],'All Labels',track_accuracy[y][2],'accuracy'])
        avg_temp.append(track_accuracy[y][2])
        

    print(' ')
    print('AVG Accuracy ',sum(avg_temp) / float(len(avg_temp)))
    avg_perfomance.append([index_scenario,x,'All Labels',sum(avg_temp) / float(len(avg_temp)),'AVG Accuracy'])
    acc_temp.append([sum(avg_temp) / float(len(avg_temp))])
    print(' ')

    avg_temp.clear()

    for y in range (0, len(track_mae)):

      if x == track_mae[y][1]:

        print(track_mae[y])
        sorted_perfomance.append([track_mae[y][0],track_mae[y][1],'All Labels',track_mae[y][2],'MAE'])
        avg_temp.append(track_mae[y][2])
        

    print(' ')
    print('AVG MAE ',sum(avg_temp) / float(len(avg_temp)))
    avg_perfomance.append([index_scenario,x,'All Labels',sum(avg_temp) / float(len(avg_temp)),'AVG MAE'])
    mae_all_temp.append([sum(avg_temp) / float(len(avg_temp))])
    print(' ')

    avg_temp.clear()

    for y in range (0, len(track_stdev)):

      if x == track_stdev[y][1]:

        print(track_stdev[y])
        sorted_perfomance.append([track_stdev[y][0],track_stdev[y][1],'All Labels',track_stdev[y][2],'STDEV'])
        avg_temp.append(track_stdev[y][2])
        

    print(' ')
    print('AVG STDEV ',sum(avg_temp) / float(len(avg_temp)))
    avg_perfomance.append([index_scenario,x,'All Labels',sum(avg_temp) / float(len(avg_temp)),'AVG STDEV'])
    stdev_temp.append([sum(avg_temp) / float(len(avg_temp))])
    print(' ')

    avg_temp.clear()

    for y in range (0, len(track_kld)):

      if x == track_kld[y][1]:

        print(track_kld[y])
        sorted_perfomance.append([track_kld[y][0],track_kld[y][1],'All Labels',track_kld[y][2],'KLD'])
        avg_temp.append(track_kld[y][2])
        

    print(' ')
    print('AVG KLD ',sum(avg_temp) / float(len(avg_temp)))
    avg_perfomance.append([index_scenario,x,'All Labels',sum(avg_temp) / float(len(avg_temp)),'AVG KLD'])
    kld_temp.append([sum(avg_temp) / float(len(avg_temp))])
    print(' ')

  report_perfomance.append([index_scenario,'All Labels','Accuracy',acc_temp])
  report_perfomance.append([index_scenario,'All Labels','MAE',mae_all_temp])
  report_perfomance.append([index_scenario,'All Labels','STDEV',stdev_temp])
  report_perfomance.append([index_scenario,'All Labels','KLD',kld_temp])

  for w in range(0,len(sorted_perfomance)):
    
    perfomance.append([index_scenario,sorted_perfomance[w]])


  print(' ')
  print('Training done in {} seconds'.format(end - start))
  print(' ')

  f.write(' \n')
  f.write('Training done in {} seconds\n'.format(end - start))
  f.write(' \n')

  

  # write_file(width, feature, batch_size)
  # exit()

if __name__ == "__main__":

  track_recall = []
  track_fmeasure = []
  track_precission = []
  track_accuracy = []
  track_mae = []
  track_stdev = []
  track_kld = []

  confusion = []

  perfomance = []
  avg_perfomance = []
  report_perfomance = []

  index_scenario = 1;

  start_s = time.time()

  scenario_code = ''.join(random.choices(string.ascii_uppercase + string.digits, k=5)) 
  scenario_desc = 'signifikansi static 15x A 111 x 200'
  
  # feature_num = [[1],[2],[3],[4],[5],[6],[7],[8],[9],[10]]
  # feature_num = [[1]]
  # feature_num = [[1,1,1,1],[1,2,3,4],[1,1,1],[1,1,1],[1,1,1,1],[1,2,3,4],[1,1,1],[1,1,1],[1,1,1,1],[1,2,3,4],[1,1,1],[1,1,1],[1,1,1,1],[1,2,3,4],[1,1,1],[1,1,1]]
  # kernel_width = [[100,100,100,100],[100,100,100,100],[100,100,100],[100,100,100],[200,200,200,200],[200,200,200,200],[200,200,200],[200,200,200],[300,300,300,300],[300,300,300,300],[300,300,300],[300,300,300],[400,400,400,400],[400,400,400,400],[400,400,400],[400,400,400]]

  feature_num = [[1,1,1],[1,1,1],[1,1,1],[1,1,1],[1,1,1],[1,1,1],[1,1,1],[1,1,1],[1,1,1],[1,1,1],[1,1,1],[1,1,1],[1,1,1],[1,1,1],[1,1,1]] #fmaps
  kernel_width = [[200,200,200],[200,200,200],[200,200,200],[200,200,200],[200,200,200],[200,200,200],[200,200,200],[200,200,200],[200,200,200],[200,200,200],[200,200,200],[200,200,200],[200,200,200],[200,200,200],[200,200,200]] #region size


  subtask_type = ["A"]

  # ts = time.time()
  # st = datetime.datetime.fromtimestamp(ts).strftime('%Y-%m-%d %H:%M:%S')
  # print(st)
  # exit()
  
  with open(r"hasil/SCENE_" + scenario_code + ".py",'w') as f:
    # mrene
    for m in range(0,len(subtask_type)):
      for n in range(0,len(feature_num)):
        
        play_scenario(feature_num[n],kernel_width[n],args.batch_size,subtask_type[m])
        index_scenario += 1

    end_s = time.time()

    print(' ')
    print('Rendering Scenario Done in {} seconds'.format(end_s - start_s))
    print(' ')

    print(' ')
    print('--- SCENARIO DESCRIPTION ---')
    print(' ')

    print(scenario_desc)

    print(' ')
    print('--- ABSOLUTELY DONE ---')
    print(' ')

    f.write(' \n')
    f.write('Rendering Scenario Done in {} seconds\n'.format(end_s - start_s))
    f.write(' \n')

    f.write(' \n')
    f.write('--- SCENARIO DESCRIPTION ---\n')
    f.write(' \n')

    f.write(scenario_desc)

    f.write(' \n')
    f.write('--- SCENARIO RESULT ---\n')
    f.write(' \n')
  
    f.write(' \n')
    f.write('--- ABSOLUTELY DONE ---\n')
    f.write(' \n')

  
  with open(r"hasil/SCENE_" + scenario_code + "_RESULT.txt",'w') as r:

      for w in range(0,len(perfomance)):
        r.write(str(perfomance[w]))
        r.write('\n')

      r.write('\n')
      r.write('--------- AVG PERFOMANCE ---------')
      r.write('\n')
      r.write('\n')

      for d in range(0,len(avg_perfomance)):
        r.write(str(avg_perfomance[d]))
        r.write('\n')

      r.write('\n')
      r.write('--------- REPORT PERFOMANCE ---------')
      r.write('\n')
      r.write('\n')

      for bb in range(0,len(report_perfomance)):
        print(report_perfomance[bb])
        r.write(str(report_perfomance[bb]))
        r.write('\n')

      n = 0
      for x in confusion:
        n = n + x

      r.write(str(n))


