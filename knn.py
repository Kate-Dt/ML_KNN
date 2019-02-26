import matplotlib.pyplot as plt
import numpy as np
import random
import math

x1 = []
x2 = []
classes = []
x1grid = []
y1grid = []
x_shift = 0
y_shift = 0
a1 = a2 = 4
b1 = 8
b2 = 9
columns = 5
rows = 8
xd = b1 - a1
yd = b2 - a2
xstep = xd / columns
ystep = yd / rows
objects = 80

def form_grid():
    x1gridNext = a1 + x_shift
    for j in range(columns + 1):
        x1grid.append(x1gridNext)
        x1gridNext += xstep
    y1gridNext = a2 + y_shift
    for j in range(rows+1):
        y1grid.append(y1gridNext)
        y1gridNext += ystep
    x2gridShift = b2 + y_shift
    for d in range(columns + 1):
        plt.plot([x1grid[d], x1grid[d]], [y_shift+a1, x2gridShift])
    y2gridShift = b1 + x_shift
    for d in range(rows+1):
        plt.plot([a1+x_shift, y2gridShift], [y1grid[d], y1grid[d]])
        
def form_objects():
    for i in range(objects):
        x1.append((b1 - a1) * random.uniform(0.0, 1.0) + x_shift + a1)
        x2.append((b2 - a2) * random.uniform(0.0, 1.0) + y_shift + a2)
    
def addClasses():
    for i in range(objects):
        column = int((x1[i] - a1) // xstep + 1)
        row = int((x2[i] - a2) // ystep + 1)
        classes.append(row*10 + column)
        
#1 - square
#2 - triangle
#3 - parabolic
#4 - superparabolic
#5 - gaussian        
def knnParzen(find_x1, find_x2, radius, kernel):
    distances = {}
    for i in range(objects):
        d = math.sqrt((find_x2 - x2[i])**2 + (find_x1 - x1[i])**2)
        if (d < radius):
            distances.update({d:classes[i]})
    sortedDistances = sorted(distances.keys())
    square = {}
    triangle = {}
    parabolic = {}
    superparabolic = {}
    gaussian = {}
    for j in range(len(sortedDistances)):
        cl = distances.get(sortedDistances[j])
        if (kernel == 1):
            if cl in square:
                square[cl] += 1
            else:
                square[cl] = 1
        elif (kernel == 2):
            r = sortedDistances[j] / radius
            w = 1 - r
            if cl in triangle:
                triangle[cl] += w
            else:
                triangle[cl] = w
        elif (kernel == 3):
            r = sortedDistances[j] / radius
            w = 1 - r**2
            if cl in parabolic:
                parabolic[cl] += w
            else:
                parabolic[cl] = w
        elif (kernel == 4):
            r = sortedDistances[j] / radius
            w = (1 - r**2)**2
            if cl in superparabolic:
                superparabolic[cl] += w
            else:
                superparabolic[cl] = w
        elif (kernel == 5):
            r = sortedDistances[j] / radius
            w = (math.e)**((-2)*r**2)
            if cl in gaussian:
                gaussian[cl] += w
            else:
                gaussian[cl] = w
    if (kernel == 1):
        sortedCount = sorted(square.values(), reverse = True)
    elif (kernel == 2):
        sortedCount = sorted(triangle.values(), reverse = True)
    elif (kernel == 3):
        sortedCount = sorted(parabolic.values(), reverse = True)
    elif (kernel == 4):
        sortedCount = sorted(superparabolic.values(), reverse = True)
    elif (kernel == 5):
        sortedCount = sorted(gaussian.values(), reverse = True)
#    print classes and weights
#    print(count)
#    print(sortedCount)
    res = -1
    if (kernel == 1):
        for cl, c in square.items():
            if c == sortedCount[0]:
                res = cl
                break
    elif (kernel == 2):    
        for cl, c in triangle.items():
            if c == sortedCount[0]:
                res = cl
                break
    elif (kernel == 3):
        for cl, c in parabolic.items():
            if c == sortedCount[0]:
                res = cl
                break
    elif (kernel == 4):
        for cl, c in superparabolic.items():
            if c == sortedCount[0]:
                res = cl
                break
    elif (kernel == 5):
        for cl, c in gaussian.items():
            if c == sortedCount[0]:
                res = cl
                break
#    crcl2 = plt.Circle((find_x1, find_x2), radius,color='black', fill=False)
#    ax2 = plt.subplot()
#    ax2.add_artist(crcl2)
#    plt.show()
    return res
        
def knn(find_x1, find_x2, k):
    distances = {}
    for i in range(objects):
        d = math.sqrt((find_x2 - x2[i])**2 + (find_x1 - x1[i])**2)
        distances.update({d:classes[i]})  
    sortedDistances = distances.keys()
    sortedDistances = sorted(sortedDistances)
    count = {}
    for j in range(k):
        cl = distances.get(sortedDistances[j])
        if cl in count:
            count[cl] += 1
        else:
            count[cl] = 1
    sortedCount = sorted(count.values(), reverse = True)
    res = -1
    for cl, c in count.items():
        if c == sortedCount[0]:
            res = cl
#    plt.scatter(find_x1, find_x2, s=80, color="red")
#    radius = sortedDistances[k]
#    crcl = plt.Circle((find_x1, find_x2), radius,color='black', fill=False)
#    ax = plt.subplot()
#    ax.add_artist(crcl)
#    plt.show()
    return res

def knnValues(find_x1, find_x2, k):
    distances = {}
    for i in range(objects):
        d = math.sqrt((find_x2 - x2[i])**2 + (find_x1 - x1[i])**2)
        distances.update({d:classes[i]})  
    sortedDistances = distances.keys()
    sortedDistances = sorted(sortedDistances)
    count = {}
    for j in range(k):
        cl = distances.get(sortedDistances[j])
        w = (k + 1 - j) / k
        if cl in count:
            count[cl] += w
        else:
            count[cl] = w
    sortedCount = sorted(count.values(), reverse = True)
#    print classes and weights
#    print(count)
#    print(sortedCount)
    res = -1
    for cl, c in count.items():
        if c == sortedCount[0]:
            res = cl
#    radius = sortedDistances[k]
#    crcl2 = plt.Circle((find_x1, find_x2), radius,color='black', fill=False)
#    ax2 = plt.subplot()
#    ax2.add_artist(crcl2)
    plt.scatter(find_x1, find_x2, s=80, color="red")
#    plt.show()
    return res

def bestK():
    countRight = 0;
    countWrong = 0;
    count = {}
    for k in range(1, objects):
        countRight = 0;
        countWrong = 0;
        for d in range(objects):
            findx = x1[d]
            findy = x2[d]
            if (knnWithoutJ(findx, findy, k, d) == classes[d]):
#                print(str(findx)+" "+str(findy) + " K:"+str(classes[d])+" knn(classes):"+str(knn(findx, findy, k)))
                countRight += 1
            else:
                countWrong += 1
        print(str(countRight) + " / " + str(countWrong) + " k: " + str(k)) 
        count[k] = countWrong   
    countSorted = sorted(count.values())
    resK = []
    print("\nBest k (wrong "+str(countSorted[0])+"): ")
    for k, wrong in count.items():
        if wrong == countSorted[0]:
            resK.append(k)
    for i in range(0, len(resK)):
        print(str(i+1)+") "+str(resK[i]))
    
def bestParzen():
    countRight = 0
    countWrong = 0
    longestRadius = 0
    if (xd > yd):
        longestRadius = xd/2
    else:
        longestRadius = yd/2
    count = []
    for i in range(1,6):
        countR = {}
        bestForRadius = 0
        for radius in np.arange(0.1, longestRadius, 0.1):
            countRight = 0;
            countWrong = 0;
            for d in range(objects):
                findx = x1[d]
                findy = x2[d]
                if (knnParzenWithoutJ(findx, findy, radius, i, d) == classes[d]):
                    countRight += 1
                else:
                    countWrong += 1
            countR.update({radius:countWrong})
        countRSorted = sorted(countR.values())
        bestVal = countRSorted[0]
        for cl,c in countR.items():
            if (c == bestVal):
                bestForRadius = cl
        count.append(bestForRadius)
    for e in range (0, len(count)):
        print("kernel: ", end="")
        print(e+1, end="; radius: ")
        print(count[e])

def knnWithoutJ(find_x1, find_x2, k, d):
    distances = {}
    for i in range(objects):
        if (i!=d):
            d = math.sqrt((find_x2 - x2[i])**2 + (find_x1 - x1[i])**2)
            distances.update({d:classes[i]})  
    sortedDistances = distances.keys()
    sortedDistances = sorted(sortedDistances)
    count = {}
    for j in range(k):
        cl = distances.get(sortedDistances[j])
        if cl in count:
            count[cl] += 1
        else:
            count[cl] = 1
    sortedCount = sorted(count.values(), reverse = True)
    res = -1
    for cl, c in count.items():
        if c == sortedCount[0]:
            res = cl
    return res

def knnParzenWithoutJ(find_x1, find_x2, radius, kernel, c):
    distances = {}
    for i in range(objects):
        if (i != c):
            d = math.sqrt((find_x2 - x2[i])**2 + (find_x1 - x1[i])**2)
            if (d < radius):
                distances.update({d:classes[i]})
    sortedDistances = sorted(distances.keys())
    square = {}
    triangle = {}
    parabolic = {}
    superparabolic = {}
    gaussian = {}
    for j in range(len(sortedDistances)):
        cl = distances.get(sortedDistances[j])
        if (kernel == 1):
            if cl in square:
                square[cl] += 1
            else:
                square[cl] = 1
        elif (kernel == 2):
            r = sortedDistances[j] / radius
            w = 1 - r
            if cl in triangle:
                triangle[cl] += w
            else:
                triangle[cl] = w
        elif (kernel == 3):
            r = sortedDistances[j] / radius
            w = 1 - r**2
            if cl in parabolic:
                parabolic[cl] += w
            else:
                parabolic[cl] = w
        elif (kernel == 4):
            r = sortedDistances[j] / radius
            w = (1 - r**2)**2
            if cl in superparabolic:
                superparabolic[cl] += w
            else:
                superparabolic[cl] = w
        elif (kernel == 5):
            r = sortedDistances[j] / radius
            w = (math.e)**((-2)*r**2)
            if cl in gaussian:
                gaussian[cl] += w
            else:
                gaussian[cl] = w
    if (kernel == 1):
        sortedCount = sorted(square.values(), reverse = True)
    elif (kernel == 2):
        sortedCount = sorted(triangle.values(), reverse = True)
    elif (kernel == 3):
        sortedCount = sorted(parabolic.values(), reverse = True)
    elif (kernel == 4):
        sortedCount = sorted(superparabolic.values(), reverse = True)
    elif (kernel == 5):
        sortedCount = sorted(gaussian.values(), reverse = True)
#    print classes and weights
#    print(count)
#    print(sortedCount)
    res = -1
    if (kernel == 1):
        for cl, c in square.items():
            if c == sortedCount[0]:
                res = cl
                break
    elif (kernel == 2):    
        for cl, c in triangle.items():
            if c == sortedCount[0]:
                res = cl
                break
    elif (kernel == 3):
        for cl, c in parabolic.items():
            if c == sortedCount[0]:
                res = cl
                break
    elif (kernel == 4):
        for cl, c in superparabolic.items():
            if c == sortedCount[0]:
                res = cl
                break
    elif (kernel == 5):
        for cl, c in gaussian.items():
            if c == sortedCount[0]:
                res = cl
                break
#    crcl2 = plt.Circle((find_x1, find_x2), radius,color='black', fill=False)
#    ax2 = plt.subplot()
#    ax2.add_artist(crcl2)
#    plt.show()
    return res

form_grid()
form_objects()
addClasses()
plt.scatter(x1, x2)

bestParzen()
#bestK()
#
#while (True):
#    x1f = float(input('Enter x1:'))
#    x2f = float(input('Enter x2:'))
#    r = float(input('Enter r:'))
##    print("\nClass: "+str(knnParzen(x1f, x2f, r, 5)))
#    print("\nClass: "+str(knnParzenWithoutJ(x1f, x2f, r, 1, 1)))
#    plt.scatter(x1, x2)
#    form_grid()
#    plt.scatter(x1f, x2f, s=80, color="red")
#    plt.show()



