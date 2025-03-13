import requests
from bs4 import BeautifulSoup
import csv
from datetime import datetime
import os
import threading
import time

def get_board_data(board):
  file_path = './data/' + board + '-titles-' + datetime.now().strftime('%Y-%m-%d') + '.csv'
  os.makedirs(os.path.dirname(file_path), exist_ok=True)
  with open(file_path, mode='w', newline='', encoding='utf-8-sig') as file:
    writer = csv.writer(file)
    writer.writerow(['push','title', 'author', 'date', 'date_details', 'link'])
    url = 'https://www.ptt.cc/bbs/' + board + '/index.html'
    counts = 0
    while True:
      data = requests.get(url)
      soup = BeautifulSoup(data.text, "html.parser")
      rows = soup.find_all('div', class_='r-ent')
      for row in rows:
          if row.find('div', class_='title').find('a') != None:
              title = row.find('div', class_='title').find('a').get_text()

              if row.find('div', class_='nrec').find('span') != None:
                push = row.find('div', class_='nrec').find('span').get_text()
              else:
                push = '0'
              
              try:
                sub_link = row.find('div', class_='title').find('a')['href']
              except:
                sub_link = ''
              
              try:
                author = row.find('div', class_='author').get_text()
              except:
                author = ''
              
              try:
                date = row.find('div', class_='date').get_text()
              except:
                date = ''

              print(push, title, author, ' '+date, sub_link)
              writer.writerow([push, title, author, ' '+date, sub_link])
              counts += 1

      while True:
        try:
          next_page = soup.find_all('a', string="‹ 上頁")[0].get("href")
          break
        except:
          pass

      if next_page == None or counts == 200000:
        break
      else:
        url =  'https://www.ptt.cc/' + next_page

    file.close()
    print(board, counts)
  return

T1 = time.perf_counter()

board_urls = [
  'https://www.ptt.cc/bbs/baseball/index.html', 
  'https://www.ptt.cc/bbs/Boy-Girl/index.html', 
  'https://www.ptt.cc/bbs/c_chat/index.html', 
  'https://www.ptt.cc/bbs/hatepolitics/index.html', 
  'https://www.ptt.cc/bbs/Lifeismoney/index.html', 
  'https://www.ptt.cc/bbs/Military/index.html', 
  'https://www.ptt.cc/bbs/pc_shopping/index.html', 
  'https://www.ptt.cc/bbs/stock/index.html', 
  'https://www.ptt.cc/bbs/Tech_Job/index.html'
]

threads = []

for i in range(len(board_urls)):
  board = board_urls[i].replace('https://www.ptt.cc/bbs/', '').replace('/index.html', '')
  threads.append(threading.Thread(target = get_board_data, args = (board, )))
  threads[i].start()

for i in range(len(board_urls)):
  threads[i].join()

T2 =time.perf_counter()

print('%s秒' % (T2 - T1))

          