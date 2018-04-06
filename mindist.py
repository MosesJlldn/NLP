import numpy as np

def levenshteinDistance(s1, s2):

    l1 = len(s1) + 1
    l2 = len(s2) + 1

    print(s1)
    print(s2)
    
    columns = []

    for y in range(l2):
        columns.append([])
        for x in range(l1):
            if (y == 0):
                columns[y].append(x)
            elif (x == 0):
                columns[y].append(y)
            else:
                columns[y].append(0)
                        
    for y in range(1,l2):
        for x in range(1,l1):
            if (s1[x - 1] == s2[y - 1]):
                columns[x][y] = columns[x - 1][y - 1]
            else:
                columns[x][y] = min (columns[x - 1][y - 1] + 2, columns[x - 1][y] + 1, columns[x][y - 1] + 1)
                
    x = len(columns) - 1
    y = len(columns[0]) - 1
    backtrace = []
    backtrace.append(columns[x][y])
    
    while (x != 0 or y != 0):
        if (s1[x - 1] == s2[y - 1]):
            backtrace.append(columns[x - 1][y - 1])
            x -= 1
            y -= 1
        else:
            minimum = min (columns[x - 1][y - 1], columns[x - 1][y], columns[x][y - 1])
            if (columns[x - 1][y - 1] == minimum):
                backtrace.append(columns[x - 1][y - 1])
                x -= 1
                y -= 1
            if (columns[x][y - 1] == minimum):
                backtrace.append(columns[x][y - 1])
                y -= 1 
            if (columns[x - 1][y] == minimum):
                backtrace.append(columns[x - 1][y])
                x -= 1
              
    for y in range(0,l2):
        print(columns[y])
    print("backtrace")
    print (backtrace)
    return ""


print(levenshteinDistance("driver", "brief"))
##print(levenshteinDistance("execution", "intention"))
