from justwatch import JustWatch
import json
just_watch = JustWatch(country="ES")
l = []
results_by_providers = just_watch.search_for_item(providers=['nfx'], page=0, content_types=['movie'], release_year_from = 2014)
for x in range(results_by_providers["total_pages"]):
    results_by_providers = just_watch.search_for_item(providers=['nfx'], page=x, content_types=['movie'], release_year_from = 2014)
    for i in results_by_providers["items"]:
        dic = dict()
        if "poster" in i:
            dic["url_image"] = "https://images.justwatch.com"+i["poster"][:-9]+"s592"
        for s in i["scoring"]:
            if s["provider_type"] == 'imdb:score':
                dic["score"] = s["value"]
                break
        dic["title"] = i["title"]
        dic["obj_type"] = i["object_type"]
        dic["id"] = i["id"]
        l.append(dic)
        
results_by_providers = just_watch.search_for_item(providers=['nfx'], page=0, content_types=['movie'], release_year_until = 2013)
for x in range(results_by_providers["total_pages"]):
    results_by_providers = just_watch.search_for_item(providers=['nfx'], page=x, content_types=['movie'], release_year_until = 2013)
    for i in results_by_providers["items"]:
        dic = dict()
        if "poster" in i:
            dic["url_image"] = "https://images.justwatch.com"+i["poster"][:-9]+"s592"
        for s in i["scoring"]:
            if s["provider_type"] == 'imdb:score':
                dic["score"] = s["value"]
                break
        dic["title"] = i["title"]
        dic["obj_type"] = i["object_type"]
        dic["id"] = i["id"]
        l.append(dic)

with open('Netflix_data.json', 'w') as json_file:
    json.dump(l, json_file)