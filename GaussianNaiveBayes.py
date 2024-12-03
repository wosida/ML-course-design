import numpy as np
from sklearn.preprocessing import StandardScaler

from Relief import Relief
from PCA import PCA

from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import KFold
from sklearn.metrics import roc_curve, auc, accuracy_score
import matplotlib.pyplot as plt

class GaussianNaiveBayes:
    def __init__(self):
        self.class_prior = {}
        self.mean = {}
        self.var = {}
        self.classes = []

    def fit(self, X, y):
        self.classes = np.unique(y) # 获取类别
        for c in self.classes:
            X_c = X[y == c] # 获取每个类别的样本
            self.class_prior[c] = len(X_c) / len(X) # 计算先验概率
            self.mean[c] = np.mean(X_c, axis=0) # 计算均值
            self.var[c] = np.var(X_c, axis=0) # 计算方差

    def _likelihood(self, x, mean, var):
        return np.exp(-(x - mean)**2 / (2 * var)) / np.sqrt(2 * np.pi * var) # 计算似然

    def _predict_instance(self, x):
        posteriors = []
        for c in self.classes:
            prior = self.class_prior[c] # 获取先验概率
            likelihood = np.sum(np.log(self._likelihood(x, self.mean[c], self.var[c]))) # 获取似然
            posterior = prior + likelihood
            posteriors.append(posterior) # 计算后验概率
        return self.classes[np.argmax(posteriors)] # 返回概率最大的类别

    def predict(self, X):
        return [self._predict_instance(x) for x in X] # 对每个样本进行预测


# 1. 数据准备
data = load_breast_cancer()
X, y = data.data, data.target
y = np.where(y == 0, -1, y)

# 2. 预处理
scaler = StandardScaler()
X = scaler.fit_transform(X)

tprs = []
aucs = []
mean_fpr = np.linspace(0, 1, 100)
accuracy_list = []
relief = Relief(n_features=25)
pca = PCA(n_components=20)

# 3. 交叉验证
kf = KFold(n_splits=10)
for train_index, test_index in kf.split(X):
    X_train, X_test = X[train_index], X[test_index]
    y_train, y_test = y[train_index], y[test_index]

    #降维和特征选择
    #X_train, X_test = relief.fit_transform(X_train, y_train), relief.transform(X_test)
    #X_train, X_test = pca.fit_transform(X_train), pca.transform(X_test)

    model = GaussianNaiveBayes()
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    # 计算并打印性能指标
    accuracy = accuracy_score(y_test, y_pred)
    accuracy_list.append(accuracy)
    fpr, tpr, thresholds = roc_curve(y_test, y_pred)
    tprs.append(np.interp(mean_fpr, fpr, tpr))
    roc_auc = auc(fpr, tpr)
    aucs.append(roc_auc)

# 绘制平均ROC曲线
mean_tpr = np.mean(tprs, axis=0)
mean_auc = auc(mean_fpr, mean_tpr)
plt.figure()
plt.plot(mean_fpr, mean_tpr, label='Mean ROC (area = %0.2f)' % mean_auc)
plt.plot([0, 1], [0, 1], 'k--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title(f'Mean ROC')
plt.legend(loc="lower right")
plt.show()

# 打印平均准确度
mean_accuracy = np.mean(accuracy_list)
print(f'Mean Accuracy: {mean_accuracy*100:.1f}%')