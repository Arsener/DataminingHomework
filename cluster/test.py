p = ['a', 'b', 'a', 'b', 'a', 'c', 'a']
pi = [0, 0, 1, 2, 3, 0, 1]
alpha = ['a', 'b', 'c']
m = len(p)
delta = [[] for i in range(m + 1)]

for a in alpha:
    if a == p[0]:
        delta[0].append(1)
    else:
        delta[0].append(0)

for q in range(1, m + 1):
    for a in range(len(alpha)):
        if q == m or p[q] != alpha[a]:
            delta[q].append(delta[pi[q - 1]][a])
        else:
            # index = -1
            # for i in range(len(alpha)):
            #     if alpha[i] == p[q - 1]:
            #         index = i
            #         break

            delta[q].append(q + 1)

for i in delta:
    print(i)

