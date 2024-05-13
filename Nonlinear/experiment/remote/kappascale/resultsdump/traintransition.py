import pickle
import sys
import matplotlib.pyplot as plt
from scipy.optimize import minimize
import numpy as np
import seaborn as sns
import optax

sys.path.append('../../../')
sys.path.append('../../../../')
from common import *
from train import train, create_train_state

mydir = sys.argv[1]
myalpha = float(sys.argv[2])

trainvals = []
testvals = []

for i in range(35):
    file_path = f'./{mydir}/pickles/train-{i}.pkl'
    with open(file_path, 'rb') as fp:
        loaded = pickle.load(fp)
    trainloss = [Metrics.loss for Metrics in loaded['train']]
    loss_array = trainloss[-1]
    trainvals.append(loss_array.item())

taus = range(11,46)

def paramrelu(xs, a, b, c):
    return [a*min(b,x)+c for x in xs]
def lossrelu(params, xs,ys):
    xs = np.array(xs)
    ys = np.array(ys)
    a,b,c = params
    yhat = np.array(paramrelu(xs,a,b,c))
    # lam = 0;
    # tb = np.where(xs==int(b)+1)[0][0]
    # flats = ys[tb:]
    return np.mean((yhat - ys)**2) #+ lam*np.mean((flats - (a*b+c))**2)
def learnrelu(xs, ys):
    result = minimize(lossrelu,[-1,25,25],args=(xs,ys))
    return result.x[0],result.x[1],result.x[2]

# trainvals = np.array(trainvals)
# plt.plot(taus,(1/trainvals)*trainvals[0],label='Final Training Error')
# plt.title(f'Train Error Transition - 2 Layers, 100 hidden dim, d = 7, k = 2*d, alpha = {myalpha}')
# a,b,c = learnrelu(taus, (1/trainvals)*trainvals[0])
# print("slope", a, "elbow", b, "offset", c)
# plt.plot(taus,paramrelu(taus,a,b,c),label='Relu Fit')
# #plt.axvline(x=b,label='Predicted Transition')
# plt.legend()
# plt.savefig(f'../plots/transition-invert-{myalpha}.png')

# def computeslope(myarr):
#     answ=[]
#     for i in range(len(myarr)):
#         if i == 0:
#             answ.append(myarr[1]-myarr[0])
#         elif i == len(myarr)-1:
#             answ.append(myarr[i]-myarr[i-1])
#         else:
#             answ.append((myarr[i+1]-myarr[i-1])/2)
#     return answ

trainvals = np.array(trainvals)
myarr = np.array(trainvals[0]/trainvals)
sums = [np.sum(myarr[0:i+1]) for i in range(len(myarr))]
threshold = 0.999; myval = threshold*sums[-1]
tstar = np.where(sums > myval)[0][0]
print(tstar)
plt.plot(taus,(1/trainvals)*trainvals[0],label='Final Training Error')
plt.axvline(x=tstar,label='Predicted Transition')
plt.legend()
plt.title(f'Train Error Transition - 2 Layers, 100 hidden dim, d = 7, k = 2*d, alpha = {myalpha}')
plt.savefig(f'../plots/transition-invert-{myalpha}.png')

# trainslopes = computeslope(trainvals)
# plt.plot(range(40),trainslopes[0:40],label='slopes approx')
# plt.axvline(x=24,label='approx max')
# plt.legend()
# plt.savefig(f'./{mydir}/testplot-{myiteration}.png')
