import numpy as np

a = np.random.rand(5,5)
file = open('file.txt', 'w')
for i in range(5):
    for j in range(5):
        file.write(str(a[i,j]))
    file.write('\n')
file.close()
print(a)