import urllib.request
import json

with open("Netflix_data.json") as f:
    data = json.load(f)

for element in data:
    if "url_image" in element:   
        r = urllib.request.urlopen(element["url_image"])
        name ="posters/"+ str(element['id'])+".jpg"
        with open(name, "wb") as f:
            f.write(r.read())