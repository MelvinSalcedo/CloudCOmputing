
import psutil
import time
import csv

row_list = [
    ["workload","Name", "Quotes"],
]

    
def cpu_utilizations():
    results = []
    i = 0
    for x in range(90):
        print (int(psutil.cpu_percent(interval=0.1)))
        x = int(psutil.cpu_percent(interval=1))
        row_list.append([x,1,5])
        results.insert(i,x)
        ++i
		#time.sleep(2)
	#print("cpu_utilization s = " , results)
    
    
    with open('1.csv', 'w', newline='') as file:
        writer = csv.writer(file, quoting=csv.QUOTE_NONNUMERIC,
                            delimiter=';')
        writer.writerows(row_list)
    return results
        
cpu_utilizations()