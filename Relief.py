import numpy as np
from typing import List, Tuple

class Relief:
    def __init__(self, n_features: int, n_neighbors: int = 1):
        self.n_features = n_features  # 要选择的特征数
        self.n_neighbors = n_neighbors  # near-miss方法中考虑的邻居数
        self.feature_scores = None  # 特征得分

    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        """
        使用输入的训练数据进行特征选择。

        Args:
            X (np.ndarray): 输入样本。
            y (np.ndarray): 样本标签。
        """
        n_samples, n_attrs = X.shape
        self.feature_scores = np.zeros(n_attrs)

        # 计算每个特征的得分
        for i in range(n_samples):
            # 找到最近的同类实例
            same_class_instances = X[y == y[i]]
            same_class_distances = np.sum((same_class_instances - X[i]) ** 2, axis=1)
            same_class_nearest = np.argmin(same_class_distances)

            # 找到最近的异类实例
            diff_class_instances = X[y != y[i]]
            diff_class_distances = np.sum((diff_class_instances - X[i]) ** 2, axis=1)
            diff_class_nearest = np.argmin(diff_class_distances)

            # 更新特征得分
            self.feature_scores -= (X[i] - same_class_instances[same_class_nearest]) ** 2
            self.feature_scores += (X[i] - diff_class_instances[diff_class_nearest]) ** 2

        self.feature_scores /= n_samples

    def transform(self,X: np.ndarray) -> np.ndarray:
        """
        使用选择的特征转换输入样本。

        Args:
            X (np.ndarray): 输入样本。

        Returns:
            np.ndarray: 转换后的样本。
        """
        # 根据特征得分排序并选择得分最高的n_features个特征
        selected_features = np.argsort(self.feature_scores)[::-1][:self.n_features]
        return X[:, selected_features]

    def fit_transform(self, X: np.ndarray, y: np.ndarray) -> np.ndarray:
        """
        使用输入的训练数据进行特征选择,并使用选择的特征转换输入样本。

        Args:
            X (np.ndarray): 输入样本。
            y (np.ndarray): 样本标签。

        Returns:
            np.ndarray: 转换后的样本。
        """
        self.fit(X, y)
        return self.transform(X)
if __name__ == "__main__":
    import numpy as np

    # 创建一个示例数据集
    np.random.seed(0)
    X = np.random.rand(100, 10)  # 100个样本，每个样本10个特征
    y = np.random.randint(0, 2, 100)  # 二分类标签

    # 创建 Relief 对象并拟合数据
    relief = Relief(n_features=5, n_neighbors=3)
    X_transformed = relief.fit_transform(X, y)

    # 输出转换后的数据形状
    print("Original shape of X:", X.shape)
    print("Transformed shape of X:", X_transformed.shape)
