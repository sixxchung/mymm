myLetter =  ['A', 'B', 'C']

i=0
for myCh in myLetter:
    print(i, myCh)
    i +=1

for i in range(len(myLetter)):
    myCh = myLetter[i]
    print(i, myCh)
# 0 A
# 1 B
# 2 C

list(enumerate(myLetter))
# [(0, 'A'), (1, 'B'), (2, 'C')]

for i, myCh in enumerate(myLetter): #, start=0):
    print(i, myCh)
# 0 A
# 1 B
# 2 C

[(i, myCh) for i , myCh in enumerate(myLetter)]