import json

with open("../duee-fin.json") as fjson,open("../dueefin2.json","w") as fout:
    for line in fjson:
        item=json.loads(line)
        events=[]
        for event in item["event_list"]:
            event["arguments"]=list(filter(lambda x:x["argument"],event["arguments"]))
            events.append(event)
        item["event_list"]=events
        fout.write(json.dumps(item,ensure_ascii=False)+"\n")