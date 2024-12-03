from sklearn.datasets import load_breast_cancer

cancer = load_breast_cancer()

print(cancer.data.shape)     #(569， 30)
print(cancer.target.shape)   #(569，)
print(cancer.target_names)   #['malignant' 'benign']
print(cancer.DESCR)