from ckip_transformers.nlp import CkipWordSegmenter, CkipPosTagger, CkipNerChunker
import csv
import os

ws_driver  = CkipWordSegmenter(model="bert-base")
pos_driver = CkipPosTagger(model="bert-base")
# ner_driver = CkipNerChunker(model="bert-base")

filter_pos = ["Caa", "Cab", "Cba", "Cbb", "P", "DE", "FW", "COLONCATEGORY","COMMACATEGORY", "DASHCATEGORY", "ETCCATEGORY", "EXCLANATIONCATEGORY", "PARENTHESISCATEGORY", "PAUSECATEGORY", "PERIODCATEGORY", "QUESTIONCATEGORY", "SEMICOLONCATEGORY", "SPCHANGECATEGORY", "PARENTHESISCATEGORY", "WHITESPACE", "EXCLAMATIONCATEGORY"]

start = 1

if os.path.exists("user-labeled-words-sample.csv"):
   with open("user-labeled-words-sample.csv", mode='r', newline='', encoding='utf-8-sig') as file:
      reader = csv.reader(file)
      token_text = list(reader)
      start = len(token_text)

base_dir = os.path.dirname(os.path.abspath(__file__))
file_path = os.path.join(base_dir, '../website/user-labeled-titles-sample.csv')

with open('user-labeled-words-sample.csv', mode='a', newline='', encoding='utf-8-sig') as write_file:
   writer = csv.writer(write_file)
   with open(file_path, mode='r', newline='', encoding='utf-8-sig') as read_file:
      reader = csv.reader(read_file)
      text = list(reader)

      for i in range(start+1, len(text)):
         token_list = []
         board = text[i][0]
         title = [text[i][1].replace("\u3000", " ")]

         if text[i][1].replace("\u3000", " ") != None and text[i][1].replace("\u3000", " ").strip() != "":
            token_list.append(board)
            ws  = ws_driver(title)
            pos = pos_driver(ws)
            # ner = ner_driver(text)

            for word_ws, word_pos in zip(ws[0], pos[0]):
               if word_pos not in filter_pos:
                  token_list.append(word_ws)

            writer.writerow(token_list)
            # print(token_list)
