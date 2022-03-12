# This is a sample Python script.

# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.

import pandas as pd
import requests
import re

def sara(id):
    url = 'https://recruitment.aimtechnologies.co/ai-tasks'
    myobj = str(id)
    x = requests.post(url, json=[myobj]).json()[myobj]
    print("***********")
    x = re.sub(r'[^\sء-ي]', '', x).split(' ')
    without_empty_strings = [string for string in x if string != ""]
    return without_empty_strings

def preprocessing():

     data =pd.read_csv("dialect_dataset.csv")
     print(data.shape)
     data['words']= data['id'].apply(sara)




# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    preprocessing()

# See PyCharm help at https://www.jetbrains.com/help/pycharm/
