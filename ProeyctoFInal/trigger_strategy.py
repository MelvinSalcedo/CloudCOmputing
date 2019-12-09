import psutil

def trigger_strategy(predictedstring, timeseriesvalue,threadup,threaddown):
    timeseriesvalue = psutil.cpu_percent(interval=1)
    print("workload = ",timeseriesvalue)
    length = len(predictedstring)
    upcount = 0
    downcount=0
    print("string predictions ",predictedstring)
    for character in predictedstring:
        
        if(ord(character)>=ord('i')):#105
            upcount= upcount + 1
            
        elif(ord(character)<=ord('h')):#104
            downcount=downcount+1
    if(upcount==length and timeseriesvalue>threadup):
        return 1
    elif(downcount==length and timeseriesvalue<threadup):
        return -1
    else:
        return 0
print (trigger_strategy('efg',5.5,4,5))
