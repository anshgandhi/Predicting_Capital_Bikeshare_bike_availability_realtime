import urllib.request
import json

station_information_url = "https://gbfs.capitalbikeshare.com/gbfs/en/station_information.json"

response = urllib.request.urlopen(station_information_url).read()
station_information_json = json.loads(str(response, 'utf-8'))

li = []
for i in station_information_json["data"]["stations"]:
    li.append([i["name"],i["station_id"],str(i["capacity"])])
li = sorted(li)

with open('./stations.csv', 'w') as f:
    for i in li:
        f.write(",".join(i)+"\n")