import numpy as np
import numpy.random as random
import matplotlib.pyplot as plt

alpha = 0.1
epsilon = 0.1
k = 10
deviation = 0.01
n_rounds = 10000
n_runs = 2000


def reward(a):
    return random.normal(q[a])
    # return q[a]


def action(Q):
    # print(random.random())
    if random.random_sample() < epsilon:
        return random.randint(0, k-1)
    else:
        return np.argmax(Q)


def step(n):
    global q

    a_sa = action(Q_sa)
    a_c = action(Q_c)

    opt_sa = a_sa == np.argmax(q)
    opt_c = a_c == np.argmax(q)

    R_sa = reward(a_sa)
    R_c = reward(a_c)

    # if n % 1000 == 0:
    #    print(R_sa, R_c)
    #    print(Q_sa, q)

    n_sa[a_sa] += 1
    Q_sa[a_sa] += 1/n_sa[a_sa] * (R_sa - Q_sa[a_sa])
    Q_c[a_c] += alpha * (R_c - Q_c[a_c])

    q += random.normal(0, deviation, size=k)

    return R_sa, R_c, opt_sa, opt_c


performance_sa = np.zeros(n_rounds)
performance_c = np.zeros(n_rounds)
optimal_sa = np.zeros(n_rounds)

for _ in range(n_runs):
    q = np.zeros(k)
    # q = random.normal(0, 1, k)
    Q_sa = np.zeros(k)
    # Q_sa = random.normal(0, 1, k)
    Q_c = np.zeros(k)
    n_sa = np.zeros(k)

    for n in range(1, n_rounds + 1):
        R_sa, R_c, opt_sa, opt_c = step(n)
        # if n == 100:
        #    print(R_sa)
        performance_sa[n - 1] += R_sa
        performance_c[n - 1] += R_c
        optimal_sa[n - 1] += opt_sa

plt.plot(range(1, n_rounds + 1), performance_sa / n_runs)
plt.plot(range(1, n_rounds + 1), performance_c / n_runs)
# plt.plot(range(1, n_rounds + 1), optimal_sa / n_runs)
plt.show()
