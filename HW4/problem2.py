import matplotlib.pyplot as plt
from math import factorial

def main():
    d = 25
    N = range(1, 101)

    C_vals = [C(n, d) for n in N]
    C_vals = [x / 2**100 for x in C_vals]

    plt.plot(N, C_vals)
    plt.show()

def C(N, d):
    vals = [ncr(N - 1, k) for k in range(d)]
    return 2 * sum(vals)

def ncr(s, k):
    result = 1
    for i in range(k):
        result *= (s - i)

    return result / factorial(k)

if __name__ == '__main__':
    main()
