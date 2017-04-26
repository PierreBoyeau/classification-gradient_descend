import numpy as np

class Bayes:
    """Pour l'instant marche uniquement pour de la classification binaire"""

    def __init__(self):
        #self.p_y = apriori.copy()

        pass

    def predict(self, data):
        func = np.array([self.log_proba(data, ind) for ind, y in enumerate(self.nb_labels)]).T
        y_predict = np.argmax(func, axis=1)
        np.place(y_predict, y_predict == 0, -1)
        return y_predict

    def log_proba(self, x, label):
        mu, std = self.mu[label], self.std[label]
        logproba = -(x - mu) ** 2 + 2 * (std ** 2)
        logproba = np.log(self.p_y[label]) + logproba
        logproba = np.sum(logproba, axis=1)
        return logproba

    def fit(self, data, labels):
        self.mu = []
        self.std = []
        self.nb_labels = np.unique(labels)
        for y in self.nb_labels:
            self.mu.append(np.mean(data[labels == y], axis=0))
            self.std.append(np.std(data[labels == y], axis=0))
        self.mu = np.array(self.mu)
        self.std = np.array(self.std)
        assert (len(self.mu) == len(self.nb_labels))
        pass

    def score(self, data, labels):
        return (self.predict(data) == labels).mean()




def optimize(func, dfunc, xinit, eps=0.1, max_iter=200):
    ind = 0
    xt = np.array(xinit)
    ft = func(xinit)
    dt = dfunc(xinit)
    X, F, D = [xt], [ft], [dt]
    for i in range(max_iter):
        xt = xt - eps*dt
        ft = func(xt)
        dt = dfunc(xt)
        X.append(xt)
        F.append(ft)
        D.append(dt)
    return np.array(X), np.array(F), np.array(D)


func = lambda x: 100 * np.power((x[1] - np.power(x[0], 2)), 2) + np.power((1 - x[0]), 2)
dfunc = lambda x: np.array([200 * (x[1] - np.power(x[0], 2)) * (-2 * x[0]) - 2 * (1 - x[0]), 200 * (x[1] - np.power(x[0], 2))])
x2, f2, d2 = optimize(func, dfunc, xinit=np.array([0,0]), eps=1e-4, max_iter=50000)


from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import numpy as np

len_grid = 50j
f = lambda x, y: 100 * np.power((y - np.power(x, 2)), 2) + np.power((1 - x), 2)
xgrid, ygrid = np.mgrid[-2:2:len_grid, -2:2:len_grid]
z = f(xgrid, ygrid)

fig = plt.figure()
ax = fig.gca(projection='3d')
ax.plot_surface(xgrid, ygrid, z)
ax.plot(x2[:,0], x2[:,1], f2)
