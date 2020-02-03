
# -*- coding: utf-8 -*-
from efficient_apriori import apriori
import fptools as fp
import csv
director = u'张艺谋'
file_name = '/Users/amy/Downloads/director/'+director+'.csv'
lists = csv.reader(open(file_name, 'r', encoding='utf-8-sig'))
# 数据加载
data = []
for names in lists:
     name_new = []
     for name in names:
           # 去掉演员数据中的空格
           name_new.append(name.strip())
     data.append(name_new[1:])
# 挖掘频繁项集和关联规则
itemsets, rules = apriori(data, min_support=0.5,  min_confidence=1)
print(itemsets)
print(rules)