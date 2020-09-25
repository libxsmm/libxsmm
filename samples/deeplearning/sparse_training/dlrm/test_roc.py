import numpy as np
from bottleneck import rankdata
import time
from roc_auc_score import roc_auc_score
import sklearn.metrics

def roc_auc_score_bottleneck(actual, predicted, approx = False):
    if approx: r = np.argsort(predicted)
    else: r = rankdata(predicted)
    n_pos = np.sum(actual)
    n_neg = len(actual) - n_pos
    sum1 = (np.sum(r[actual==1]) - n_pos*(n_pos+1)/2)
    print(f"bottleneck nPos {n_pos}  nNeg {n_neg}, sum {sum1}")
    return sum1 / (n_pos*n_neg)

d = np.load('targets_scores.npz')
t = d['targets'].reshape([-1])
s = d['scores'].reshape([-1])

#t = t[:2000000]
#s = s[:2000000]
# t = np.array([0.0, 1.0, 1.0, 0.0, 1.0, 0.0, 0.0, 1.0, 0.0, 0.0]).astype(np.float32)
# s = np.array([0.1, 0.3, 0.7, 0.2, 0.1, 0.3, 0.4, 0.9, 0.2, 0.7]).astype(np.float32)
# t = np.array([0.0, 1.0, 1.0, 0.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0]).astype(np.float32)
# s = np.array([0.1, 0.1, 0.2, 0.2, 0.3, 0.3, 0.4, 0.7, 0.7, 0.9]).astype(np.float32)
#              #[1.5  1.5  3.5  3.5  5.5  5.5  7.0  8.5  8.5  10.0
#              #[     0     1              2         3         4
print(f"# pairs: {t.shape}")
roc = roc_auc_score(t, s)
t1 = time.time()
roc = roc_auc_score(t, s)
t2 = time.time()
print(f"{'C++:':12s} roc: {roc} Time: {t2-t1:.4f} sec")

t1 = time.time()
roc = roc_auc_score(t, s)
t2 = time.time()
print(f"{'C++:':12s} roc: {roc} Time: {t2-t1:.4f} sec")

t1 = time.time()
#roc = sklearn.metrics.roc_auc_score(t, s)
acc = sklearn.metrics.accuracy_score(t, np.round(s))
t2 = time.time()
#print(f"{'sklearn:':12s} roc: {roc:.7f} Time: {t2-t1:.4f} sec")
print(f"{'sklearn acc:':12s} acc: {acc:.7f} Time: {t2-t1:.4f} sec")

t1 = time.time()
loss = sklearn.metrics.log_loss(t, s)
t2 = time.time()
print(f"{'sklearn:':12s} loss: {loss:.7f} Time: {t2-t1:.4f} sec")

t1 = time.time()
roc = roc_auc_score_bottleneck(t, s)
t2 = time.time()
print(f"{'bottleneck:':12s} roc: {roc:.7f} Time: {t2-t1:.4f} sec")


