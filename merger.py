import json

data2 = json.load(open('TrueFactsStated-2017.json'))
data3 = json.load(open('TrueFactsStated-2018.json'))

all_data = data2 + data3
with open('all.json', 'w') as outfile:
    json.dump(all_data, outfile)