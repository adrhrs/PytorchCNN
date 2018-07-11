import emot
import html
import os
import re

class ReadDataTopik:
    def __init__(self, file_name):
        self.file_name = file_name
        self.emoticons = {}
        self.emojis    = {}

    def replace_URL(self, string):
        tokens  = ['<url>' if '://' in token else token
                  for token in string.split()]
        return ' '.join(tokens)
  
    def replace_mention(self, string):
        tokens  = ['<mention>' if token.startswith('@') else token
                   for token in string.split()]
        return ' '.join(tokens)
  
    def replace_mult_occurences(self, string):
        return re.sub(r'(.)\1{2,}', r'\1\1', string)
  
    def replace_token_emoticon(self, token):
        if token.startswith('<3'):
          return ' <hati> '
        if token.startswith(':'):
          if token == ':D':
            return ' <tertawa> '
          if (token == ':P') | (token == ':p'):
            return ' <menjulurkan_lidah> '
          if (token == ':O') | (token == ':o'):
            return ' <terkejut> '
          if (token == ':x') | (token == ':*'):
            return ' <cium> '
          if token == ':3':
            return ' <malu-malu_kucing> '
        if token.startswith('='):
          if (token == '=/') | (token == '=\\'):
            return ' <terganggu> '
        if token == 'XD':
          return ' <tertawa_terbahak-bahak> '
        return token
  
    def replace_emoticons(self, string):
        # Senyum
        string = string.replace(':))', ' <senyum_senyum> ')
        string = string.replace(':)', ' <senyum> ')
        string = string.replace(':-))', ' <senyum_senyum> ')
        string = string.replace(':-)', ' <senyum> ')
        string = string.replace('((:', ' <senyum_senyum> ')
        string = string.replace('(:', ' <senyum> ')
        string = string.replace('((-:', ' <senyum_senyum> ')
        string = string.replace('(-:', ' <senyum> ')
        string = string.replace('=))', ' <senyum_senyum> ')
        string = string.replace('=)', ' <senyum> ')
        string = string.replace('^_^', ' <senyum> ')

        # Sedih
        string = string.replace(':((', ' <sedih_sedih> ')
        string = string.replace(':(', ' <sedih> ')
        string = string.replace(':-((', ' <sedih_sedih> ')
        string = string.replace(':-(', ' <sedih> ')
        string = string.replace(')):', ' <sedih_sedih> ')
        string = string.replace('):', ' <sedih> ')
        string = string.replace('))-:', ' <sedih_sedih> ')
        string = string.replace(')-:', ' <sedih> ')

        # Berkedip
        string = string.replace(';))', ' <senyum_berkedip> ')
        string = string.replace(';)', ' <senyum_berkedip> ')

        # Tears
        string = string.replace(":'))", ' <menangis_bahagia> ')
        string = string.replace(":')", ' <menangis_bahagia> ')
        string = string.replace(":'((", ' <menangis_sedih> ')
        string = string.replace(":'(", ' <menangis_sedih> ')
        string = string.replace("((':", ' <menangis_bahagia> ')
        string = string.replace("(':", ' <menangis_bahagia> ')

        # Some annoyed
        string = string.replace(':/', ' <terganggu> ')
        string = string.replace(':\\', ' <terganggu> ')

        # Straight face
        string = string.replace(':|', ' <muka_datar> ')
        string = string.replace(':-|', ' <muka_datar> ')

        string = ' '.join([self.replace_token_emoticon(token)
                            for token in string.split()])
        return string
  
    def clean_str(self, string):
        string = re.sub(r"[^A-Za-z0-9(),!?\'\`]", " ", string)
        string = re.sub(r"[^A-Za-z]+", " ", string)
        string = re.sub(r"\'s", " \'s", string)
        string = re.sub(r"\'ve", " \'ve", string)
        string = re.sub(r"n\'t", " n\'t", string)
        string = re.sub(r"\'re", " \'re", string)
        string = re.sub(r"\'d", " \'d", string)
        string = re.sub(r"\'ll", " \'ll", string)
        string = re.sub(r",", " , ", string)
        string = re.sub(r"!", " ! ", string)
        string = re.sub(r"\(", " \( ", string)
        string = re.sub(r"\)", " \) ", string)
        string = re.sub(r"\?", " \? ", string)
        string = re.sub(r"\s{2,}", " ", string)
        return string.strip()
      
    def divide_line(self, line):
        tokens      = line.split()
        id_tweet    = tokens[0]
        topik       = tokens[1]
        lbl_sntmnt  = tokens[2]
        lbl_topik   = tokens[3]
        lbl_sarkas  = tokens[4]
        message     = ' '.join(tokens[5:]).strip('"')
        return id_tweet, topik, lbl_sntmnt, lbl_topik, lbl_sarkas, message

    def getAngka(self):
        return 3

    def __iter__(self):

        with open(self.file_name, 'r', encoding='utf8', errors='ignore') as f:
            
            for line in f:
                id_tweet, topik, lbl_sntmnt, lbl_topik, lbl_sarkas, message = self.divide_line(line)
                message = self.replace_URL(message)
                message = html.unescape(message)
                message = message.replace('""', ' <kutip> ')
                message = self.replace_mention(message)
                message = self.replace_mult_occurences(message)
                message = message.replace('..', ' <elipsis> ')
                message = self.replace_emoticons(message)
                message = self.clean_str(message)
                message = self.replace_mention(message)
                message = message.lower()

                # print(lbl_sntmnt)
                
                yield id_tweet, lbl_sntmnt, message