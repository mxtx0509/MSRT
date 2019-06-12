from matplotlib import pyplot as plt  
import argparse
import os
import sys
filename = sys.argv[1]
reader = open(filename)
loss=[]
for line in reader:
    if line.split(' ')[0]=='Epoch:':
        temp=line.split('Loss ')[1].split(' (')[0]
        if float(temp)>1.5:
            temp = 1.5
            print (temp)
        loss.append(float(temp))
print (loss[1])
plt.plot(loss) 
plt.savefig('plot1.png', format='png') 
plt.show() 