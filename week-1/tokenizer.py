from ckip_transformers.nlp import CkipWordSegmenter, CkipPosTagger, CkipNerChunker
import csv

ws_driver  = CkipWordSegmenter(model="bert-base")
pos_driver = CkipPosTagger(model="bert-base")
# ner_driver = CkipNerChunker(model="bert-base")

filter_pos = ["Caa", "Cab", "Cba", "Cbb", "P", "DE", "FW", "COLONCATEGORY","COMMACATEGORY", "DASHCATEGORY", "ETCCATEGORY", "EXCLANATIONCATEGORY", "PARENTHESISCATEGORY", "PAUSECATEGORY", "PERIODCATEGORY", "QUESTIONCATEGORY", "SEMICOLONCATEGORY", "SPCHANGECATEGORY", "PARENTHESISCATEGORY", "WHITESPACE", "EXCLAMATIONCATEGORY"]

with open('data-clean-words-sample.csv', mode='w', newline='', encoding='utf-8-sig') as write_file:
   writer = csv.writer(write_file)
   with open("data-clean-sample.csv", mode='r', newline='', encoding='utf-8-sig') as read_file:
      reader = csv.reader(read_file)
      text = list(reader)
      for i in range(1, len(text)):
         token_list = []
         board = text[i][0]
         token_list.append(board)
         title = [text[i][4].replace("\u3000", " ")]
         ws  = ws_driver(title)
         pos = pos_driver(ws)
         # ner = ner_driver(text)

         for word_ws, word_pos in zip(ws[0], pos[0]):
            if word_pos not in filter_pos:
               token_list.append(word_ws)

         writer.writerow(token_list)
         # print(token_list)
         