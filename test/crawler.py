import requests
from bs4 import BeautifulSoup
import csv
from datetime import datetime
import os
import threading
import time

def save_to_csv(path, data_patch):
    with open(path, mode='a', newline='', encoding='utf-8-sig') as file:
        writer = csv.writer(file)
        writer.writerows(data_patch)

def get_board_data(url, path):
  with open(path, mode='r', newline='', encoding='utf-8-sig') as file:
    reader = csv.reader(file)
    counts = len(list(reader))
  while True:
    data_patch = []
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
              text_link = row.find('div', class_='title').find('a')['href']
            except:
              text_link = ''
            
            try:
              author = row.find('div', class_='author').get_text()
            except:
              author = ''
            
            try:
              date = row.find('div', class_='date').get_text()
            except:
              date = ''

            # print(push, title, author, ' '+date, text_link, url)
            data_patch.append([push, title, author, ' '+date, text_link, url])
            counts += 1

    save_to_csv(path, data_patch)
    print(url)

    next_page = soup.find_all('a', string="‹ 上頁")[0].get("href") 
    if next_page == None or counts == 200000:
      print(url, "已經抓到20萬筆或第一頁了")
      break
    else:
      url =  'https://www.ptt.cc/' + next_page

  return

T1 = time.perf_counter()

urls = {
  'baseball':'https://www.ptt.cc/bbs/baseball/index.html', 
  'Boy-Girl':'https://www.ptt.cc/bbs/Boy-Girl/index.html', 
  'c_chat':'https://www.ptt.cc/bbs/c_chat/index.html', 
  'hatepolitics':'https://www.ptt.cc/bbs/hatepolitics/index.html', 
  'Lifeismoney':'https://www.ptt.cc/bbs/Lifeismoney/index.html', 
  'Military':'https://www.ptt.cc/bbs/Military/index.html',
  'pc_shopping':'https://www.ptt.cc/bbs/pc_shopping/index.html', 
  'stock':'https://www.ptt.cc/bbs/stock/index.html', 
  'Tech_Job':'https://www.ptt.cc/bbs/Tech_Job/index.html'
}

threads = []

folder = './data/'
count = 0
if not os.path.exists(folder):
  os.mkdir(folder)
  for board, url in urls.items():
    path = './data/' + board + '-titles-' + datetime.now().strftime('%Y-%m-%d') + '.csv'
    with open(path, mode='w', newline='', encoding='utf-8-sig') as file:
      writer = csv.writer(file)
      writer.writerow(['push','title', 'author', 'date', 'text_link', 'url'])

    threads.append(threading.Thread(target = get_board_data, args = (url, path)))
    threads[count].start()
    count += 1
else:
  for items in os.listdir(folder):
    path = os.path.join(folder, items)
    with open(path, mode='r', newline='', encoding='utf-8-sig') as file:
      reader = csv.reader(file)
      last_url = list(reader)[-1][-1]
      last_page = int(last_url.split('index')[1].replace('.html', ''))
      if last_page == 1:
        print(items, "已經抓到第一頁了")
      else:
        url = last_url.split('index')[0] + 'index' + str(last_page - 1) + '.html'
        threads.append(threading.Thread(target = get_board_data, args = (url, path)))
        threads[count].start()
        count += 1

for i in range(count):
  threads[i].join()

T2 =time.perf_counter()

print('%s秒' % (T2 - T1))

          