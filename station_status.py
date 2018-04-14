import MySQLdb
import urllib
import urllib.request
import json
import datetime

datetime.datetime.now().strftime("%Y-%m-%d")
station_status_url = "https://gbfs.capitalbikeshare.com/gbfs/en/station_status.json"
station_information_url = "https://gbfs.capitalbikeshare.com/gbfs/en/station_information.json"

response = urllib.request.urlopen(station_status_url).read()
station_status_json = json.loads(str(response, 'utf-8'))

response = urllib.request.urlopen(station_information_url).read()
station_information_json = json.loads(str(response, 'utf-8'))
station_to_id = {}

for i in station_information_json["data"]["stations"]:
    station_to_id[i["station_id"]] = [i["capacity"]]

timestamp = [int(x) for x in datetime.datetime.fromtimestamp(station_status_json["last_updated"]).strftime("%m-%d-%H-%M").split("-")]
data = []
for i in station_status_json["data"]["stations"]:
    data.append((*(timestamp),int(i["station_id"]),*(station_to_id[i["station_id"]]),i["num_docks_available"],i["num_bikes_available"]))
db=MySQLdb.connect(host="35.226.98.33",user="ansh",passwd="pass@123",db="stations_data")
cursor = db.cursor()
query = "INSERT INTO agg VALUES (%s,%s,%s,%s,%s,%s,%s,%s)"
for i in data:
    cursor.execute(query,i)
    db.commit()
db.close()
