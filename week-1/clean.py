import csv
import os


  writer.writerow(['board', 'count', 'push', 'category','title', 'author', 'month', 'day', 'text_link'])

  folder = './data/'
  for items in os.listdir(folder):
    path = os.path.join(folder, items)
    with open(path, mode='r', newline='', encoding='utf-8-sig') as file:
      reader = csv.reader(file)
      data = list(reader)
      count = 0
      for i in range(1, len(data)):
        if count < 200000:
          title = data[i][1].strip().lower()
          if title.startswith('re:') or title.startswith('fw:'):
            pass
          else:
            count += 1
            board = data[i][4].split('/')[2]
            push = data[i][0]
            category = ""
            if title.startswith('['):
              category = title.split('[')[1].split(']')[0]
            author = data[i][2].strip().lower()
            month = data[i][3].split('/')[0].strip()
            try:
              day = data[i][3].split('/')[1].strip()
            except:
              day = ""
            text_link = data[i][4].strip()
            writer.writerow([board, count, push, category, title, author, month, day, text_link])
        else:
          break
      print(items, count, "完成")

# baseball-titles-2025-03-14.csv 200000 完成
# Boy-Girl-titles-2025-03-14.csv 56294 完成
# c_chat-titles-2025-03-14.csv 200000 完成
# hatepolitics-titles-2025-03-14.csv 64355 完成
# Lifeismoney-titles-2025-03-14.csv 69257 完成
# Military-titles-2025-03-14.csv 36245 完成
# pc_shopping-titles-2025-03-14.csv 71027 完成
# stock-titles-2025-03-14.csv 127257 完成
# Tech_Job-titles-2025-03-14.csv 59043 完成
        