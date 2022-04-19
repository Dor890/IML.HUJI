from IMLearn.metrics.loss_functions import mean_square_error
import numpy as np

q = int(input("Enter a q\n"))
if q == 3:
    x = str(input("Enter a string\n\r"))
    pal = ""
    for i in range(len(x)):
        for j in range(len(x), i, -1):
            if len(pal) >= j - i:
                break
            elif x[i:j] == x[i:j][::-1]:
                pal = x[i:j]
                break
    print(pal)
else:
    pass