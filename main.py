import pandas as pd
import requests
import re
import aiohttp
import asyncio
import nest_asyncio
from joblib import logger

nest_asyncio.apply()

def preprocessing():

     data =pd.read_csv("data/external/dialect_dataset.csv")
     n = int(data.shape[0]/1000)  # chunk row size
     list_df = [data[i:i + n] for i in range(0, data.shape[0], n)]
     # print(data.shape)
     print(len(list_df))
     return list_df


async def main(df):
    x=0
    async with aiohttp.ClientSession() as session:
        tasks = []
        for index, row in df.iterrows():
            task = asyncio.ensure_future(get_data(session, row['id']))
            tasks.append(task)
            print(x)
            x+=1
        return_data = await asyncio.gather(*tasks)
    return return_data

async def get_data(session, id):
    url = 'https://recruitment.aimtechnologies.co/ai-tasks'
    myobj = str(id)
    async with session.post(url, json=[myobj]) as response:
        try:
          result_data = (await response.json())[myobj]
        except requests.Timeout as err:
          logger.error({"message": err.message})
        # print(result_data)
        result =  re.sub(r'[^\sء-ي]', '', str(result_data))
        return result

def begin():
  list_of_df=preprocessing()
  for i in range(len(list_of_df)):
    output=asyncio.run(main(list_of_df[i]))
    list_of_df[i]["text"]=output
    list_of_df[i].to_csv(r'data/processed/processed{}.csv'.format(i), index=False)

if __name__== "__main__":
  begin()
