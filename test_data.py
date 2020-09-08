import pandas as pd
import random
import sys

print(sys.argv)
if sys.argv[1]:
    df = pd.read_csv("./training.csv", header=None)
    data = df.values
    for index in range(10):
        if sys.argv[1] == '1':
            num = random.randint(0, 34)
            data[index, num] = 0 if data[index, num] == 1 else 0
        if sys.argv[1] == '2':
            num1, num2 = random.randint(0, 34), random.randint(0, 34)
            data[index, num1] = 0 if data[index, num1] == 1 else 0
            data[index, num2] = 0 if data[index, num2] == 1 else 0
        if sys.argv[1] == '3':
            num1, num2, num3 = random.randint(0, 34), random.randint(
                0, 34), random.randint(0, 34)
            data[index, num1] = 0 if data[index, num1] == 1 else 0
            data[index, num2] = 0 if data[index, num2] == 1 else 0
            data[index, num3] = 0 if data[index, num3] == 1 else 0

    df2 = pd.DataFrame(data)
    df2.to_csv(f'test_{sys.argv[1]}.csv', index=False, header=False)
