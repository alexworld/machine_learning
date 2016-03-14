import pandas

data = pandas.read_csv('titanic.csv', index_col = 'PassengerId')
A = dict()
cnt1 = 0
cnt2 = 0

for name in data['Name']:
    print(name)
    name = name[name.find(',') + 1 :].strip()
    gender = name[: name.find('.')].strip()
    name = name[name.find('.') + 1 :].strip()
    name = name + ' '

    if name.find('(') == -1:
        name = name[: name.find(' ')]
    else:
        name = name[name.find('(') + 1: ].strip()
        name = name[: name.find(' ')]

   
    name = name.replace('\"', '')
    name = name.replace('\'', '')
    name = name.replace('(', '')
    name = name.replace(')', '')

    if name in ['Mary']:
        cnt1 += 1
    if name in ['Anna']:
        cnt2 += 1
 
    
    print(gender, name)

    if gender[-1] == 's':
        if name in A:
            A[name] += 1
        else:
            A[name] = 1
res = []
print('MATCH', cnt1, cnt2)

for i in A:
  res.append((A[i], i))
res.sort(reverse = True)
print(res)
