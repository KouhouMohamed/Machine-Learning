# -*- coding: utf-8 -*-
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_regression
from sklearn.linear_model import SGDRegressor


#pour des valeur aléatoire
np.random.seed(0)

#n_samples:nbr d'echantillons, n_feat:nombre des var, noise:bruit
# créer les matrices x et y aléatoirement
x, y = make_regression(n_samples=100, n_features=1, noise=10)

#visualiser les pts (x,y)
plt.scatter(x,y)


model = SGDRegressor(max_iter=100,eta0=0.01)
model.fit(x,y)
print("Coeff R2 = ",model.score(x,y))
#plt.scatter(x,y)
plt.plot(x,model.predict(x),c='red',lw=3)
