from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn import tree
from sklearn.metrics import accuracy_score
import graphviz

# 准备手写数字数据集
digits = datasets.load_digits()
# 获取特征和标识
features = digits.data
labels = digits.target
# 选取数据集的33%为测试集，其余为训练集
train_features, test_features, train_labels, test_labels = train_test_split(features, labels, test_size=0.33)
# 创建CART分类树
clf = tree.DecisionTreeClassifier()
# 拟合构造CART分类树
clf.fit(train_features, train_labels)
# 预测测试集结果
test_predict = clf.predict(test_features)
# 测试集结果评价
print('CART分类树准确率:', accuracy_score(test_labels, test_predict))
# 画决策树
dot_data = tree.export_graphviz(clf, out_file=None)
graph = graphviz.Source(dot_data)
graph.render('CART//CART_practice_digits')
