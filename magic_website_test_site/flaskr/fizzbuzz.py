import time



print('hello, welcome to fizz_buzz!')

count = 0 
check = True

while(check is True):
    count = count + 1
    print('Please give your answer!')
    starttime = time.time()
    value = input().lower()
    lasttime = time.time()
    if(count%3 == 0):
        if(count%5 == 0):
            correct_value = 'fizzbuzz'
        else:
            correct_value = 'fizz'
    elif(count%5 == 0):
        correct_value = 'buzz'
    else:
        correct_value = f'{count}'
    if str(value) == correct_value and round(lasttime-starttime) < 5:
        print("Ding!")
    else:
        print("Bzz!")
        check = False                