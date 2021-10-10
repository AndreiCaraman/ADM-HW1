# Say "Hello, World!" With Python

if __name__ == '__main__':
    print "Hello, World!"

# Python If-Else

import math
import os
import random
import re
import sys


if __name__ == '__main__':
    n = int(raw_input().strip())
if n % 2 != 0:
     print("Weird")
elif (n >= 2 and n <= 5):
    print("Not Weird")
elif (n >= 6 and n <= 20):
     print("Weird")
else:
    print("Not Weird")

# Arithmetic Operators

if __name__ == '__main__':
    a = int(input())
    b = int(input())


print(a+b, a-b, a*b, sep='\n')

# Python: Division

if __name__ == '__main__':
    a = int(input())
    b = int(input())


print(a//b, a/b, sep='\n')

# Loops

if __name__ == '__main__':
    n = int(input())

for i in range(n):
     print(i**2, sep='\n')

# Write a function

def is_leap(year):
    leap = False

    if year % 4 == 0:
        leap = True
        if year % 100 == 0:
            leap = False
            if year % 400 == 0:
                leap = True

    return leap

# Print Function

if __name__ == '__main__':
    n = int(input())

i = 1
while i < (n + 1):
    print(i, end='')
    i += 1

# List Comprehensions

if __name__ == '__main__':
    x = int(input())
    y = int(input())
    z = int(input())
    n = int(input())

v = [[i, j, k] for i in range(x + 1) for j in range(y + 1) for k in range(z + 1) if ((i + j + k) != n)]

print(v)

# Find the Runner-Up Score!

if __name__ == '__main__':
    n = int(input())
    arr = map(int, input().split())

l = list(arr)
champ = max(l)
while max(l) == champ:
    l.remove(max(l))
print(max(l))

# Nested Lists

records = list()

if __name__ == '__main__':
    for i in range(int(input())):
        name = input()
        score = float(input())
        records.append([name, score])

lowest_grade = min(records, key=lambda record: record[1])

while min(records, key=lambda record: record[1]) == lowest_grade:
    records.remove(min(records, key=lambda record: record[1]))

second_lowest_grade = min(records, key=lambda record: record[1])

records.sort(key=lambda record: record[1])

second_lower_grade_students = []
i = 0
while records[i][1] == second_lowest_grade[1]:
        second_lower_grade_students.append(records[i])
        i += 1

second_lower_grade_students.sort()
for j in range(len(second_lower_grade_students)):
    print(second_lower_grade_students[j][0])

# Finding the percentage

if __name__ == '__main__':
    n = int(input())
    student_marks = {}
    for _ in range(n):
        name, *line = input().split()
        scores = list(map(float, line))
        student_marks[name] = scores
    query_name = input()

avg = sum(student_marks[query_name])/len(student_marks[query_name])


print("{:.2f}".format(avg))

# Lists

if __name__ == '__main__':
    N = int(input())
    l = []
    for _ in range(N):
        s = input().split()
        command = s[0]
        arguments = s[1:]
        if command != "print":
            command += "(" + ",".join(arguments) + ")"
            eval("l." + command)
        else:
            print(l)

# Tuples

if __name__ == '__main__':
    n = int(raw_input())
    integer_list = map(int, raw_input().split())

t = tuple(integer_list)
print(hash(t))

# sWAP cASE

def swap_case(s):
    return s.swapcase()




if __name__ == '__main__':
    s = input()
    result = swap_case(s)
    print(result)

# String Split and Join

def split_and_join(line):
    line = line.split(" ")
    return '-'.join(line)

if __name__ == '__main__':
    line = input()
    result = split_and_join(line)
    print(result)

# What's Your Name?

def print_full_name(first, last):
    name = first + " " + last
    print("Hello ", name, "! You just delved into python.", sep="")
if __name__ == '__main__':
    first_name = input()
    last_name = input()
    print_full_name(first_name, last_name)

# Mutations

def mutate_string(string, position, character):
    l = list(string)
    l[position] = character
    string = ''.join(l)
    return string

if __name__ == '__main__':
    s = input()
    i, c = input().split()
    s_new = mutate_string(s, int(i), c)
    print(s_new)

# Find a string

def count_substring(string, sub_string):
    occurances = int()
    lsb = len(sub_string)
    i = 0

    while len(string[i:(lsb + i)]) == lsb:
        if string[i:(lsb + i)] == sub_string:
            occurances += 1
        i += 1
    return occurances

if __name__ == '__main__':
    string = input().strip()
    sub_string = input().strip()

    count = count_substring(string, sub_string)
    print(count)

# String Validators

if __name__ == '__main__':
    s = input()

n = len(s)

alphanumeric = False
alphabetical = False
digits = False
lowercase = False
uppercase = False

i=0
while alphanumeric == False and i < n:
    if s[i].isalnum():
        alphanumeric = True
    else:
        i += 1
i=0
while alphabetical == False and i < n:
    if s[i].isalpha():
        alphabetical = True
    else:
        i += 1

i=0
while digits == False and i < n:
    if s[i].isdigit():
        digits = True
    else:
        i += 1
i=0
while lowercase == False and i < n:
    if s[i].islower():
        lowercase = True
    else:
        i += 1
i=0
while uppercase == False and i < n:
    if s[i].isupper():
        uppercase = True
    else:
        i += 1

print(alphanumeric, alphabetical, digits, lowercase, uppercase, sep="\n")

# Text Alignment

#Replace all ______ with rjust, ljust or center.

thickness = int(input()) #This must be an odd number
c = 'H'

#Top Cone
for i in range(thickness):
    print((c*i).rjust(thickness-1)+c+(c*i).ljust(thickness-1))
#Top Pillars
for i in range(thickness+1):
    print((c*thickness).center(thickness*2)+(c*thickness).center(thickness*6))
#Middle Belt
for i in range((thickness+1)//2):
    print((c*thickness*5).center(thickness*6))
#Bottom Pillars
for i in range(thickness+1):
    print((c*thickness).center(thickness*2)+(c*thickness).center(thickness*6))
#Bottom Cone
for i in range(thickness):
    print(((c*(thickness-i-1)).rjust(thickness)+c+(c*(thickness-i-1)).ljust(thickness)).rjust(thickness*6))

# Text Wrap

import textwrap

def wrap(string, max_width):
    s= ''
    for i in range(0, len(string), max_width):
        s=s + string[i:i+max_width]+"\n"
    return s

if __name__ == '__main__':
    string, max_width = input(), int(input())
    result = wrap(string, max_width)
    print(result)

# Designer Door Mat

n, k = map(int , input("").split(" "))
st ='.|.'
st1='-'
avg = int((n+1)/2)
for i in range(1,1+n):
    if avg==i:
        print("WELCOME".center(k,st1))
    elif i<avg:
        print((st*(2*i-1)).center(k,st1))
    elif avg<i:
        print((st*(2*(n-i)+1)).center(k,st1))

# Map and Lambda Function

cube = lambda x: x**3 # complete the lambda function

def fibonacci(n):
    F = [0, 1]
    i = 2
    while i < n:
        F.append(F[i - 2] + F[i - 1])
        i += 1

    return F[:n]

if __name__ == '__main__':
    n = int(input())
    print(list(map(cube, fibonacci(n))))

# String Formatting

def print_formatted(number):
    lenght = len(str(bin(number))[2:])
    for i in range(1,number+1):
        d = str(i).rjust(lenght)
        o = str(oct(i))[2:].rjust(lenght)
        h = str(hex(i))[2:].rjust(lenght).upper()
        b = str(bin(i))[2:].rjust(lenght)
        print(d, o, h, b)


if __name__ == '__main__':
    n = int(input())
    print_formatted(n)

# Capitalize!

def solve(s):
    s  = s.rsplit()
    S = [i.capitalize() for i in s]
    return " ".join(S)


if __name__ == '__main__':
    fptr = open(os.environ['OUTPUT_PATH'], 'w')

    s = input()

    result = solve(s)

    fptr.write(result + '\n')

    fptr.close()

# Introduction to Sets

def average(array):

    a = sum(set(array))/len(set(array))
    return f"{a: .3f}"

if __name__ == '__main__':
    n = int(input())
    arr = list(map(int, input().split()))
    result = average(arr)
    print(result)

# Symmetric Difference

M = int(input())
a = set(map(int, input().split()))
N = input()
b = set(input())

print(a)

# Set .add()

s = set()
N = int(input())
for _ in range(N):
    s.add(input())

print(len(s))

# Set .discard(), .remove() & .pop()

n = int(input())
s = set(map(int, input().split()))
c = int(input())


def op(s, c):

    for _ in range(c):
        k = input().split()
        if len(k) == 2:
            eval("s." + k[0] + "(" + k[1] + ")")
        else:
            s.pop()

    return print(sum(s))


op(s, c)

# Set .union() Operation

n_a = int(input())
a = set(map(int, input().split()))
n_b = int(input())
b = set(map(int, input().split()))

print(len(a.union(b)))

# Set .intersection() Operation

n=int(input())

a=set(map(int, input().split()))

m=int(input())

b=set(map(int, input().split()))

print(len(a.intersection(b)))


# Set .difference() Operation

set1  = set()
set2 = set()
n = int(input())
for i in range(0,n):
    roll = input()
    set1.add(roll)
F = int(input())
for j in range(0,F):
    roll1 = input()
    set2.add(roll1)
diff = len(set1-set2)
print(diff)


# Set .symmetric_difference() Operation

_, a = input(), set(input().split())
_, b = input(), set(input().split())
print(len(a.symmetric_difference(b)))

# Set Mutations

n = int(input(""))
seta = set(map(int , input("").split(" ")))
j = int(input(""))
for i in range(j):
    ian = list(input().split())
    if ian[0]== "intersection_update":
        fi = set(int(x) for x in input('').split(" "))
        seta.intersection_update(fi)

    elif ian[0] == "symmetric_difference_update":
        fi = set(int(x) for x in input('').split(" "))
        seta.symmetric_difference_update(fi)

    elif ian[0] == "update":
        fi = set(int(x) for x in input('').split(" "))
        seta.update(fi)

    elif ian[0] == "difference_update":
        fi = set(int(x) for x in input('').split(" "))
        seta.difference_update(fi)
res = 0
for m in seta:
    res+=m
print(res)

# The Captain's Room

K = int(input())
t = list(map(int, input().split())) # total of elements
L = len(t) # n of people
f = (L - 1) / K # n of families
s = set() # single room
m = set() # multiple rooms

for i in range(L):
    if t[i] not in s:
        s.add(t[i])
    else:
        m.add(t[i])

print(s.difference(m).pop())

# Check Subset

T = int(input())
for _ in range(T):
    nA = int(input())
    A = set(map(int, input().split()))
    nB = int(input())
    B = set(map(int, input().split()))
    print(A.intersection(B) == A)

# Check Strict Superset

A = set(map(int, input().split()))
n = int(input())
B1 = set(map(int, input().split()))
B2 = set(map(int, input().split()))

print((((A | B1) ^ B1) == 1) & (((A | B2) ^ B2) == 1))

# Zipped!

NX = list(map(int, input().split())) #students, subjects
S = [list(map(float, input().split())) for i in range(NX[1])]

g = list(zip(*S)) # grades for the i-th student

for i in range(NX[0]):
    avg = (sum(g[i]) / NX[1])
    print(round(avg, 1))

# Arrays

import numpy

def arrays(arr):
    v = numpy.array(arr[::-1], float)
    return v

arr = input().strip().split(' ')

# Shape and Reshape

import numpy

v = numpy.array(list(map(int, input().split())))
v.shape = (3, 3)
print(v)


# Concatenate

import numpy

NMP = list(map(int, input().split()))
N_matrix = []
for i in range(NMP[0]):
    N_matrix.append(list(map(int, input().split())))

M_matrix = []
for j in range(NMP[1]):
    M_matrix.append(list(map(int, input().split())))

print(numpy.concatenate((N_matrix, M_matrix), axis=0))

# Transpose and Flatten

import numpy

NM = list(map(int, input().split()))
l = list()
for _ in range(NM[0]):
    l.append(list(map(int, input().split())))

m = numpy.array(l)

print(numpy.transpose(m))
print(m.flatten())

# Zeros and Ones

import numpy


NML = tuple(map(int, input().split()))

print(numpy.zeros(NML, dtype = numpy.int0))
print(numpy.ones(NML, dtype = numpy.int0))

# Eye and Identity

import numpy

numpy.set_printoptions(legacy='1.13')

n,m = input().split()

i = numpy.eye(int(n), int(m))
print(i)

# Array Mathematics

import numpy

import numpy as np

nRow,mCol=tuple(map(int,input().split()))

a=np.array([input().split()],dtype=int)

b=np.array(input().split(),dtype=int)

print(a+b,a-b,a*b,a//b,a%b,a**b,sep='\n')


# Floor, Ceil and Rint

import numpy

numpy.set_printoptions(legacy="1.13")
my_array = numpy.array(input().split(),float)
print(numpy.floor(my_array))
print(numpy.ceil(my_array))
print(numpy.rint(my_array))

# Sum and Prod

l=[]
a,b =map(int,input().split())
for i in range(a):
    l.append(list(map(int, input().split())))

print(numpy.prod(numpy.sum(l, axis=0)))

# Min and Max

import numpy as np
N, M = map(int, input().strip().split())
my_array = np.array([ input().strip().split() for _ in range(N) ], int)
print(np.max(np.min(my_array, axis = 1)))


# Mean, Var, and Std

import numpy

n,m=map(int,input().split())
a=numpy.array([input().split() for i in range(n)],int)
print(numpy.mean(a,axis=1))
print(numpy.var(a,axis=0))
print(numpy.around(numpy.std(a,axis=None),11))

# Dot and Cross

import numpy

n = int(input())
a = numpy.array([input().split() for _ in range(n)], int)
b = numpy.array([input().split() for _ in range(n)], int)
print(numpy.dot(a, b))

# Inner and Outer

import numpy

a =[int(i) for i in input().split()]
b =[int(i) for i in input().split()]
A = numpy.array(a)
B = numpy.array(b)
print(numpy.inner(A, B))
print(numpy.outer(A, B))

# Polynomials

import numpy

print(numpy.polyval(list(map(float, input().split())),int(input())))

# Linear Algebra

import numpy as np
np.set_printoptions(legacy='1.13')

n=int(input())
a=np.array([input().split() for _ in range(n)],float)

print(np.linalg.det(a))

# Alphabet Rangoli

def print_rangoli(size):
    width= (size+(size-1)+2*(size-1))
    n=size
    for i in range(n):
        s=''
        ch=96+n
        mid= ((2*i+1)//2)
        for j in range(2*i+1):
            s+=chr(ch)
            if j<mid:
                ch-=1
            else:
                ch+=1

            if j<(2*i):
                s+='-'
            else:
                pass
        print(s.center(width,'-'))
    for i2 in range(1,n):
        s=''
        ch=96+n
        mid= ((1+2*(n-1-i2))//2)
        for j2  in range(1+2*(n-1-i2)):
            s+=chr(ch)
            if j2<mid:
                ch-=1
            else:
                ch+=1

            if j2<(2*(n-1-i2)):
                s+='-'
            else:
                pass

        print(s.center(width,'-'))
if __name__ == '__main__':
    n = int(input())
    print_rangoli(n)

# The Minion Game


def minion_game(string):
    Stuart = 0
    Kevin = 0
    vowels = 'AEIOU'
    l = len(string)
    for i in range(l):
        if string[i] in vowels:
           Kevin += l - i
        else:
           Stuart += l - i
    if Stuart > Kevin:
        print(f"Stuart {Stuart}")
    elif Stuart < Kevin:
        print(f"Kevin {Kevin}")
    else:
        print("Draw")


if __name__ == '__main__':
    s = input()
    minion_game(s)

# Merge the Tools!

def merge_the_tools(string, k):
    for i in range(k,len(string)+1,k):
        d = []
        for q in string[i-k:i]:
            if q not in d:
                d.append(q)
        print("".join(d))

if __name__ == '__main__':
    string, k = input(), int(input())
    merge_the_tools(string, k)

# No Idea!

n, m = map(int, input().split())
v = list(map(int, input().split()))
A = set(map(int, input().split()))
B = set(map(int, input().split()))
h = int()
for i in v:
    if i in A:
        h += 1
    elif i in B:
        h -= 1

print(h)

# Athlete Sort

#!/bin/python3

import math
import os
import random
import re
import sys



if __name__ == '__main__':
    nm = input().split()

    n = int(nm[0])

    m = int(nm[1])

    arr = []

    for _ in range(n):
        arr.append(list(map(int, input().rstrip().split())))

    k = int(input())

arr.sort(key=lambda x: x[k])
for i in arr:
    print(*i)

# collections.Counter()

from collections import Counter

#variables--------------------------------------------------------
n_of_shoes = int(input())
shoe_stock = Counter(list(map(int, input().split())))
n_of_customers = int(input())
customers_needs = [list(map(int, input().split())) for _ in  \\
    range(n_of_customers)] # list of pairs shoe size, price
money = int()

#calculations-----------------------------------------------------
for customer in customers_needs:
    if customer[0] in shoe_stock.keys():
        money += customer[1]
        shoe_stock[customer[0]] -= 1
        if shoe_stock[customer[0]] == 0:
            del shoe_stock[customer[0]]

print(money)

# Collections.namedtuple()

nStudents = int(input())
marks_index = input().split().index('MARKS')
avg = int()

for _ in range(nStudents):
    avg += int(input().split()[marks_index])

print(f'{avg/nStudents: .2f}' )

# Collections.OrderedDict()

from collections import OrderedDict

nItems = int(input())
consumers_shopping = OrderedDict()
for _ in range(nItems):
    I = input().split()
    item_name = ' '.join(I[:-1])
    price = int(I[-1])
    if item_name not in consumers_shopping:
        consumers_shopping[item_name] = price
    else:
        net_price = consumers_shopping[item_name]
        consumers_shopping[item_name] = net_price + price

for items in consumers_shopping:
    print(items, consumers_shopping[items])

# Collections.deque()

from collections import deque
d = deque()
nOperations = int(input())
for _ in range(nOperations):
    I = input()
    if I.isalpha():
        eval('d.' + I + '()')
    else:
        cmd, value = I.split()
        eval('d.' + cmd + '(' + value + ')')

print(*d)

# Word Order

from collections import OrderedDict

n_words = int(input())
words = OrderedDict()
for _ in range(n_words):
    word = input()
    words[word] = words.get(word, 0) + 1

print(len(words))
print(*words.values())

# Company Logo

#!/bin/python3

import math
import os
import random
import re
import sys

from collections import Counter

if __name__ == '__main__':
    s = input()

letters = Counter(sorted([i for i in s]))
for letter, occurrence in letters.most_common(3):
    print(letter, occurrence)

# Piling Up!

from collections import deque

n_test_cases = int(input())
for _ in range(n_test_cases):
    n_cubes = int(input())
    l = deque(map(int, input().split()))
    stacked_cubes = list()
    if l[0] <= l[-1]:
        stacked_cubes.append(l.pop())
    else:
        stacked_cubes.append(l.popleft())

    for i in range(1, n_cubes):
        if l[0] <= l[-1] and l[-1] <= stacked_cubes[i - 1]:
            stacked_cubes.append(l.pop())
        elif l[0] > l[-1] and l[0] <= stacked_cubes[i - 1]:
            stacked_cubes.append(l.popleft())
        else:
            print('No')
            break
    if len(stacked_cubes) == n_cubes:
        print('Yes')

# DefaultDict Tutorial

from collections import defaultdict

n, m = map(int, input().split())
A = defaultdict(list)
B = defaultdict(list)
# A = {input() for i in range(n)}
for i in range(1, n + 1): # inserimento valori nel dizionario
    A[input()].append(i)
for i in range(1, m + 1):
    B[input()].append(i)

for k in B.keys(): #print dei valori comuni
    if k in A:
        print(*A[k])
    else:
        print(-1)

# ginortS

s = input()
lower, upper, even, odd = [], [], [], []

for i in s:
    if i.islower():
        lower.append(i)
    elif i.isupper():
        upper.append(i)
    elif int(i) % 2 == 0:
        even.append(i)
    else:
        odd.append(i)
lower.sort()
upper.sort()
even.sort()
odd.sort()
lower = ''.join(lower)
upper = ''.join(upper)
even = ''.join(even)
odd = ''.join(odd)

print(lower, upper, odd, even, sep='')

# Calendar Module

import calendar
I = list(map(int, input().split()))
d = calendar.weekday(I[2], I[0], I[1])
print(calendar.day_name[d].upper())

# Time Delta

def time_delta(t1, t2):
    from datetime import datetime as dt
    format = '%a %d %b %Y %H:%M:%S %z'
    t1 = dt.strptime(t1, format)
    t2 = dt.strptime(t2, format)
    delta = abs((t1 - t2).total_seconds())
    # print(int(abs((dt.strptime(input(), format) - dt.strptime(input(), format)).total_seconds())))
    return int(delta)

t = int(input())
for _ in range(t):
    t1 = input()
    t2 = input()
    print(time_delta(t1, t2))

# Exceptions

for _ in range(int(input())):
    try:
        a, b = map(int, input().split())
        print(a//b)
    except (ZeroDivisionError, ValueError) as e:
        print(f"Error Code: {e}")

# Standardize Mobile Number Using Decorators

def wrapper(f):
    def fun(l):
        return f(f"+91 {x[-10:-5]} {x[-5:]}" for x in l)
    return fun

@wrapper
def sort_phone(l):
    print(*sorted(l), sep='\n')

if __name__ == '__main__':
    l = [input() for _ in range(int(input()))]
    sort_phone(l)


# Decorators 2 - Name Directory

import operator

def person_lister(f):
    def inner(people):
        return (f(person) for person in sorted(people, key=lambda x: int(x[2])))
    return inner

@person_lister
def name_format(person):
    return ("Mr. " if person[3] == "M" else "Ms. ") + person[0] + " " + person[1]

if __name__ == '__main__':
    people = [input().split() for i in range(int(input()))]
    print(*name_format(people), sep='\n')

# Re.split()

regex_pattern = r"[.,]"	# Do not delete 'r'.

import re
print("\n".join(re.split(regex_pattern, input())))

# Detect Floating Point Number

import re
def isfloat(f):
        try:
            float(f)
            return True
        except ValueError:
            return False

pattern = re.compile(r"([+-.]|\d)\.\d")
def test(I):
    print(bool(re.search(pattern, I)) and isfloat(I)
    )

for _ in range(int(input())):
    test(input())

# Validating phone numbers

import re
pattern = re.compile(r"^[789]\d\d\d\d\d\d\d\d\d$")
for _ in range(int(input())):
    N = input()
    # if len(N) != 10 or not N.isnumeric():
    answer = "NO"
    # elif
    if bool(re.match(pattern, N)) == True:
        answer = "YES"
    print(answer)

# Validating and Parsing Email Addresses

import email.utils, re
pat = re.compile(r'^[a-zA-Z][a-zA-Z\.\-\_0-9]+@[a-zA-Z]+\.[a-zA-Z]{1,3}$')

for _ in range(int(input())):
    I = input()
    m = email.utils.parseaddr(I)[1]
    if bool(re.search(pat, m)):
        print(I)

# Hex Color Code

import re
patt = re.compile(r'#[0-9A-Fa-f]{3,6}')
inbrackets = False
for _ in range(int(input())):
    I = input()
    if re.search(r'{', I):
        inbrackets = True
        continue
    elif re.search(r'}', I):
        inbrackets = False
        continue
    else:
        while inbrackets:
            matches = patt.finditer(I)
            for i in matches:
                print(i.group(0), sep="\n")
            break

# Group(), Groups() & Groupdict()

import re
m = re.search(r'([a-zA-Z0-9])\1', input())
print(m.group(1) if m else -1)

# Re.findall() & Re.finditer()

import re
I = input()
pat = re.compile(r'([qwrtypsdfghjklzxcvbnm])([aeiou]{2,})([qwrtypsdfghjklzxcvbnm])', flags=re.IGNORECASE)
matches = re.finditer(pat, I)

if len(re.findall(pat, I)) != 0:
    for match in matches:
        print(match.group(2))
else:
    print(-1)

#
