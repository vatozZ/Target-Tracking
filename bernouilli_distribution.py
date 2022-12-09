"""
Implementation of Bernouilli Distribution
p : success probability (also expectation of Bernouilli distribution)
q : 1-p -> failure probability
n_events describe the how many experiments are going to happen.

@VatozZ
"""

import numpy as np
from numpy.random import rand
import matplotlib.pyplot as plt

p = 0.8 # success probability
#q = 1-p # failure probability
n_events = 10

class BernouilliDistribution:
    def __init__(self, p=0.5, size=1):
        self.p = p
        self.q = self.p
        self.size = size
        self.rv_outcomes = []

    def _rvs(self):

        for i in range(0, self.size):
            random_var = np.random.rand()
            print("random_var", random_var)
            if random_var <= self.p:
                self.rv_outcomes.append(1)
            else:
                self.rv_outcomes.append(0)
        return self.rv_outcomes

    def pmf(self):
        return self.rv_outcomes

    def mean(self):
        return self.p

    def variance(self):
        return self.p * self.q

    def show_pmf(self):
        plt.figure()
        plt.xlim(-1, 2)
        n_success = np.sum(np.array(self.rv_outcomes), axis=0)
        n_fail = np.shape(np.array(self.rv_outcomes))[0] - n_success
        plt.title('Bernouilli Probability Mass Function')
        plt.stem(0, n_fail, linefmt='C0-', label='n-success')
        plt.stem(1, n_success, linefmt='C1-', label='n-fail')
        plt.annotate(text=str(n_success), xy=(1.05, n_success))
        plt.annotate(text=str(n_fail), xy=(0.05, n_fail))
        plt.legend()
        plt.show()


bern = BernouilliDistribution(p=p, size=n_events)
pmf = bern._rvs()
plt.title('Bernouilli Distributed Outcomes of Experiment with n='+str(n_events)+' events')
plt.stem(pmf, label='outcomes')
plt.legend()
plt.show()
bern.show_pmf()


