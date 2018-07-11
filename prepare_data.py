from gensim.models.keyedvectors import KeyedVectors
import numpy as np
import random
import re
import time
import torchtext
from read_data import ReadDataTopik

class prepare_data:

    def __init__(self):
        self.emb_init_value = None
        self.vocab_to_idx   = None
        self.idx_to_vocab   = None
        self.label_to_idx   = None
        self.idx_to_label   = None
        self.embed_num      = 0
        self.label_num      = 0
        self.train_set      = None
        self.dev_set        = None
        self.idq            = 0
  
    def polarity_to_label(self, polarity):
    # Di sini musti ditambain lagi untuk subtask A dan B
        
        # print(polarity)

        if int(polarity) == 1:
            label = 'strong negative'
        elif int(polarity) == 2:
            label = 'negative'
        elif int(polarity) == 4:
            label = 'positive'
        elif int(polarity) == 5:
            label = 'strong positive'
        else:
            label = 'neutral'

        self.idq += 1
        return label

    def read_dataset(self,type,subtask):
        
        if type == 'indo':
            # print('Reading Indonesian')
            if subtask == 'C':
            	file_name = 'Data/all_new2.txt'
            elif subtask == 'B':
            	file_name = 'Data/all_new_32.txt'
            elif subtask == 'A':
                file_name = 'Data/all_new_duality.txt'
        else:
            # print('Reading MR')
            file_name = 'Data/subj.txt'

        indo_ecommerce_data = ReadDataTopik(file_name)

        # print('selesai baca')
        # idq = 0

        twt_id_field = torchtext.data.Field(use_vocab=False, sequential=False)
        label_field = torchtext.data.Field(sequential=False)
        text_field = torchtext.data.Field()

        fields = [('twt_id', twt_id_field), ('label', label_field), ('text', text_field)]

        # asd = 0
        # for twt_id, polarity, text in indo_ecommerce_data:
        #     asd += 1
        #     print(asd,polarity)

        # exit()
        
        self.fields = fields
        
        examples = [
                        torchtext.data.Example.fromlist([twt_id, self.polarity_to_label(polarity), text], fields) 

                        for twt_id, polarity, text in indo_ecommerce_data

                    ]
                        
        self.examples = examples

        # print([(example.label) for example in examples])

        # for e in range(0,len(self.examples)):
        #     t = self.examples.label
        #     print(t)


        # print('selesai masukin array')

        # exit()

        # print(len(self.examples))

        # exit()

 

    def create_fold_embeddings(self, embeddings, args, key_vector):
    
        emb_init_values = []
        unk = []

        a = 0
        b = 0

        if args.embeddings_source == 'none':
            for i in range(self.idx_to_vocab.__len__()):  # Untuk memastikan bahwa urut
                word = self.idx_to_vocab.get(i)
                if word == '<unk>':
                    emb_init_values.append(np.random.uniform(-0.25, 0.25, args.embeddings_dim).astype('float32'))

                elif word == '<pad>':
                    emb_init_values.append(np.zeros(args.embeddings_dim).astype('float32'))
                    
                elif word in key_vector.wv.vocab:
                    emb_init_values.append(key_vector.wv.word_vec(word))
                    b = b+1
                else:
                    emb_init_values.append(np.random.uniform(-0.25, 0.25, args.embeddings_dim).astype('float32'))
                    a = a+1
                    unk.append(word)
                    # print(word)
                
        self.emb_init_values = emb_init_values

        known_word = b
        unknown_word = a

        print(known_word, unknown_word)


        

        return known_word, unknown_word, emb_init_values