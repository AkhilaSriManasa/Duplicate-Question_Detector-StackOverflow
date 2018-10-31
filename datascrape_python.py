import re
import csv
import requests
from bs4 import BeautifulSoup
data_list=[]
for i in range(1,21):
    site = requests.get('https://stackoverflow.com/questions/tagged/python?page="+str(i)+"&sort=newest&pagesize=50');
    if site.status_code is 200:
        soup= BeautifulSoup(site.text, 'html.parser')
        questions = soup.find_all(class_='question-summary')
        for question in questions:
            topic = question.find(text=re.compile('[duplicate]'),class_='question-hyperlink').get_text()
            url =   question.find(class_='question-hyperlink').get('href')
            #views = question.find(class_='views').find(class_='mini-counts').find('span').get_text()
            ques = requests.get("https://stackoverflow.com"+url);
            if ques.status_code is 200:
                soup= BeautifulSoup(ques.text, 'html.parser')
                #questions = soup.find_all(class_='question-summary')
                body_content= soup.find(class_='post-text').get_text()
            #votes = question.find(class_='votes').find(class_='mini-counts').find('span').get_text()
            new_data = {"PYTHON": topic,"url": url, "body_content":body_content}
            data_list.append(new_data)
        with open ('tag_python.csv','w') as file:
            writer = csv.DictWriter(file, fieldnames = ["PYTHON","url","body_content"])            
            writer.writeheader()
            for row in data_list:
                writer.writerow(row)
