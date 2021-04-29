import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC, LinearSVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import Perceptron
from sklearn.linear_model import SGDClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import cross_val_score, KFold
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from xgboost import XGBClassifier

train = pd.read_csv('/python/titanic/train.csv')
test = pd.read_csv('/python/titanic/test.csv')
total = [train, test]

# print(train, test, sep='\n')
# print(train.isnull().sum(), test.isnull().sum(), sep='\n')

# train_num = train[['Age', 'SibSp', 'Parch', 'Fare']]
# train_cat = train[['Pclass', 'Sex', 'Ticket', 'Cabin', 'Embarked']]

# train[['SibSp', 'Survived']].groupby(['SibSp'], as_index=False).mean().sort_values(by='Survived', ascending=False)
# train[['Parch', 'Survived']].groupby(['Parch'], as_index=False).mean().sort_values(by='Survived', ascending=False)
# train[['Pclass', 'Survived']].groupby(['Pclass'], as_index=False).mean().sort_values(by='Survived', ascending=False)
# train[['Sex', 'Survived']].groupby(['Sex'], as_index=False).mean().sort_values(by='Survived', ascending=False)

# Pclass가 3인 승객들의 생존율이 낮았음을 알수 있다.
# grid = sns.FacetGrid(train, row='Pclass', col='Survived')
# grid.map(plt.hist, 'Age', alpha=0.5, bins=20)
# plt.show()

# pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)

# Ticket을 제거합니다.
for data in total:
    data.drop(['Ticket'], axis=1, inplace=True)

# Cabin을 알파벳 첫번째 글자로 구분하고 Pclass 기준으로 중간값으로 결측치를 채웁니다.
for data in total:
    data['Cabin'] = data['Cabin'].str[:1]
    data['Cabin'] = data['Cabin'].map({'A': 0, 'B': 1, 'C': 2, 'D': 3, 'E': 4, 'F': 5, 'G': 6, 'T': 7})
    data['Cabin'].fillna(data.groupby('Pclass')['Cabin'].transform('median'), inplace=True)

for data in total:
    data.drop(['Cabin'], axis=1, inplace=True)

# Name에서 Title으로 추출해서 구분한다.
for data in total:
    data['Title'] = data['Name'].str.extract(' ([A-Za-z]+)\.', expand=False)

# sns.countplot(data=train, x='Title', hue='Survived')
# print(pd.crosstab(train['Title'], train['Sex']))

for data in total:
    data['Title'] = data['Title'].replace(['Lady', 'Countess', 'Capt', 'Col',\
        'Don', 'Dr', 'Major', 'Rev', 'Sir', 'Jonkheer', 'Dona'], 'others')
    data['Title'] = data['Title'].replace('Mlle', 'Miss')
    data['Title'] = data['Title'].replace('Ms', 'Miss')
    data['Title'] = data['Title'].replace('Mme', 'Mrs')

# print(train[['Title', 'Survived']].groupby(['Title'], as_index=False).mean())

for data in total:
    data['Title'] = data['Title'].astype('category').cat.codes
    data.drop(['Name'], axis=1, inplace=True)

# Fare 결측치를 채우고 분위수 기준으로 구간을 나눈다.
for data in total:
    data['Fare'].fillna(data.groupby('Pclass')['Fare'].transform('median'), inplace=True)
    data['Fare'] = pd.qcut(data['Fare'], 10, labels=False)

# Age의 결측치를 Title을 기준으로 중간값으로 채운다.
for data in total:
    data['Age'].fillna(data.groupby('Title')['Age'].transform('median'), inplace=True)
    
# Sex를 0과 1로 나눕니다.
for data in total:
    data['Sex'] = data['Sex'].map({'male': 0, 'female': 1}).astype(int)

# Age를 구간을 임의로 나눠서 분류한다.
for data in total:
    data['Age'] = pd.cut(data['Age'], [0, 10, 15, 20, 25, 30, 35, 40, 45, 50, 60, 100], labels=False)

# 본인이 속한 가족구성원의 수를 FamilySize에 저장한다.
for data in total:
    data['FamilySize'] = data['SibSp'] + data['Parch'] + 1

# print(train[['FamilySize', 'Survived']].groupby(['FamilySize'], as_index=False)\
#         .mean().sort_values(by='Survived', ascending=False))

# 핵가족인 경우는 1, 아니면 0이다.
for data in total:
    data['Nuclear'] = 0
    data.loc[(2 <= data['FamilySize']) & (data['FamilySize'] <= 4), 'Nuclear'] = 1

# 불필요해진 특성인 FamilySize는 지운다.
for data in total:
    data.drop('FamilySize', axis=1, inplace=True)

# print(train[['IsAlone', 'Survived']].groupby(['IsAlone'], as_index=False).mean())

# 출발항 결측치는 가장 많은 출발항 S로 채운다.
train['Embarked'].fillna(train['Embarked'].mode()[0], inplace=True)

for data in total:
    data['Embarked'] = data['Embarked'].astype('category').cat.codes

X_train = train.drop(['PassengerId', 'Survived'], axis=1)
Y_train = train['Survived']
X_test = test.drop('PassengerId', axis=1).copy()

# print(X_train, Y_train, X_test, sep='\n')

cv = KFold(n_splits=10, shuffle=True, random_state=1)

# 다양한 머신러닝 모델을 적용해서 교차 검증 점수로 순위를 매긴다.
perceptron = Perceptron(random_state=1)
perceptron.fit(X_train, Y_train)
score_perceptron = round(cross_val_score(perceptron, X_train, Y_train, cv=cv).mean() * 100, 3)
print('Perceptron의 교차 검증 점수 : {}.'.format(score_perceptron))

gnb = GaussianNB()
gnb.fit(X_train, Y_train)
score_gnb = round(cross_val_score(gnb, X_train, Y_train, cv=cv).mean() * 100, 3)
print('GaussainNB의 교차 검증 점수 : {}'.format(score_gnb))

logreg = LogisticRegression(max_iter=2000, random_state=1)
logreg.fit(X_train, Y_train)
score_logreg = round(cross_val_score(logreg, X_train, Y_train, cv=cv).mean() * 100, 3)
print('로지스틱회귀의 교차 검증 점수 : {}'.format(score_logreg))

linear_svc = LinearSVC(max_iter=10000, random_state=1)
linear_svc.fit(X_train, Y_train)
score_linear_svc = round(cross_val_score(linear_svc, X_train, Y_train, cv=cv).mean() * 100, 3)
print('LinearSVC의 교차 검증 점수 : {}'.format(score_linear_svc))

svc = SVC(random_state=1)
svc.fit(X_train, Y_train)
score_svc = round(cross_val_score(svc, X_train, Y_train, cv=cv).mean() * 100, 3)
print('SVC의 교차 검증 점수 : {}'.format(score_svc))

knn = KNeighborsClassifier()
knn.fit(X_train, Y_train)
score_knn = round(cross_val_score(knn, X_train, Y_train, cv=cv).mean() * 100, 3)
print('KNN의 교차 검증 점수 : {}'.format(score_knn))

decision_tree = DecisionTreeClassifier(random_state=1)
decision_tree.fit(X_train, Y_train)
score_decision_tree = round(cross_val_score(decision_tree, X_train, Y_train, cv=cv).mean() * 100, 3)
print('결정트리의 교차 검증 점수 : {}'.format(score_decision_tree))

random_forest = RandomForestClassifier(n_estimators=50, max_depth=1, max_features=3, random_state=1)
random_forest.fit(X_train, Y_train)
score_random_forest = round(cross_val_score(random_forest, X_train, Y_train, cv=cv).mean() * 100, 3)
print('랜덤포레스트의 교차 검증 점수 : {}'.format(score_random_forest))

gbc = GradientBoostingClassifier(max_depth=1, max_features=3, random_state=1)
gbc.fit(X_train, Y_train)
score_gbc = round(cross_val_score(gbc, X_train, Y_train, cv=cv).mean() * 100, 3)
print('그래디언트부스팅분류의 교차 검증 점수 : {}'.format(score_gbc))

mlp = MLPClassifier(max_iter=1000, random_state=1)
mlp.fit(X_train, Y_train)
score_mlp = round(cross_val_score(mlp, X_train, Y_train, cv=cv).mean() * 100, 3)
print('MLP의 교차 검증 점수 : {}'.format(score_mlp))

sgd = SGDClassifier(random_state=1)
sgd.fit(X_train, Y_train)
score_sgd = round(cross_val_score(sgd, X_train, Y_train, cv=cv).mean() * 100, 3)
print('SGD의 교차 검증 점수 : {}'.format(score_sgd))

models = pd.DataFrame([['Perceptron', score_perceptron, perceptron],\
                      ['GaussianNB', score_gnb, gnb], ['KNN', score_knn, knn],\
                      ['Random Forest', score_random_forest, random_forest],\
                      ['Decision Tree', score_decision_tree, decision_tree],\
                      ['Gradient Boosting Classifier', score_gbc, gbc],\
                      ['MLP', score_mlp, mlp], ['Linear SVC', score_linear_svc, linear_svc],\
                      ['Stochastic Gradient Decent', score_sgd, sgd],\
                      ['Logistic Regression', score_logreg, logreg], ['SVC', score_svc, svc]],\
                    columns=['Model', 'Score', 'Estimator'])
sorted_models = models.sort_values(by=['Score'], axis=0, ascending=False, ignore_index=True).loc[:,['Model','Score']]
best_estimator = models.sort_values(by=['Score'], axis=0, ascending=False, ignore_index=True).loc[:,['Estimator','Score']]
print(sorted_models)
print('교차 검증 점수가 가장 높은 모델은 {}이고 점수는 {}이다'.format(sorted_models.iloc[0, 0], sorted_models.iloc[0, 1]))
Y_pred = best_estimator.iloc[0, 0].predict(X_test)

# 그리드서치나 랜덤서치로 하이퍼파라미터를 세부적으로 조정한다.
# svc = SVC(random_state=1)
# parameters = {'C': [0.001, 0.01, 0.1, 1, 10, 100],\
#               'gamma': [0.001, 0.01, 0.1, 1, 10, 100],\
#               'degree': [1, 2, 3, 4, 5, 6]}
# grid_svc = GridSearchCV(svc, param_grid=parameters, cv=cv, n_jobs=-1, verbose=3)
# grid_svc.fit(X_train, Y_train)
# score_grid_svc = pd.DataFrame(grid_svc.cv_results_)
# print(score_grid_svc[['params', 'mean_test_score', 'rank_test_score',\
#                            ]])
# print('SVC의 그리드서치 최고 모델: {}'.format(grid_svc.best_estimator_))
# print('SVC의 그리드서치 최고 파라미터: {}'.format(grid_svc.best_params_))
# print('SVC의 그리츠서치 최고 정확도: {0:.3f}'.format(round(grid_svc.best_score_ * 100, 3)))
# estimator = grid_svc.best_estimator_
# Y_pred = estimator.predict(X_test)

# adaboost = AdaBoostClassifier(random_state=1)
# parameters = {'n_estimators': [50, 100, 150, 200, 300],\
#               'learning_rate': [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1]}
# grid_adaboost = GridSearchCV(adaboost, param_grid=parameters, cv=Kfold, n_jobs=-1, verbose=3)
# grid_adaboost.fit(X_train, Y_train)
# score_grid_adaboost = pd.DataFrame(grid_adaboost.cv_results_)
# print(score_grid_adaboost[['params', 'mean_test_score', 'rank_test_score',\
#                            ]])
# print('AdaBoost의 그리드서치 최고 모델: {}'.format(grid_adaboost.best_estimator_))
# print('AdaBoost의 그리드서치 최고 파라미터: {}'.format(grid_adaboost.best_params_))
# print('AdaBoost의 그리츠서치 최고 정확도: {0:.3f}'.format(round(grid_adaboost.best_score_ * 100, 3)))
# estimator = grid_adaboost.best_estimator_
# Y_pred = estimator.predict(X_test)

# xgb = XGBClassifier(objective='binary:logistic', eval_metric='logloss',\
#                     silent=1, random_state=1)
# parameters = {'n_estimators': [10, 20, 50, 100, 200, 500], 'max_depth': [1, 2, 3, 4, 5],\
#               'learing_rate': [0.001, 0.01, 0.1], 'subsample': [0.5, 0.6, 0.7, 0.8, 0.9, 1]}
# grid_xgb = GridSearchCV(xgb, param_grid=parameters, cv=Kfold, n_jobs=-1)
# grid_xgb.fit(X_train, Y_train, early_stopping_rounds=100, eval_set= [(X_train, Y_train)])
# score_grid_xgb = pd.DataFrame(grid_xgb.cv_results_)
# print(score_grid_xgb[['params', 'mean_test_score', 'rank_test_score',\
#                            ]])
# print('XGB의 그리드서치 최고 모델: {}'.format(grid_xgb.best_estimator_))
# print('XGB의 그리드서치 최고 파라미터: {}'.format(grid_xgb.best_params_))
# print('XGB의 그리츠서치 최고 정확도: {0:.3f}'.format(round(grid_xgb.best_score_ * 100, 3)))
# estimator = grid_xgb.best_estimator_
# Y_pred = estimator.predict(X_test)

# random_forest = RandomForestClassifier(random_state=1)
# parameters = {'max_depth': [1, 2, 3, 4, None], 'max_features': [1, 2, 3, 4, 'auto', 'sqrt', 'log2'],\
#               'n_estimators': [20, 50, 100],
#               'min_samples_leaf': [1, 2, 3, 4],\
#               'min_samples_split': [2, 3, 4]}
# grid_random_forest = GridSearchCV(random_forest, param_grid=parameters, cv=Kfold, n_jobs=-1, verbose=3)
# grid_random_forest.fit(X_train, Y_train)
# score_random_forest = pd.DataFrame(grid_random_forest.cv_results_)
# print(score_random_forest[['params', 'mean_test_score', 'rank_test_score',\
#                            ]])
# print('랜덤 포레스트의 그리드서치 최고 모델: {}'.format(grid_random_forest.best_estimator_))
# print('랜덤 포레스트의 그리드서치 최고 파라미터: {}'.format(grid_random_forest.best_params_))
# print('랜덤 포레스트의 그리츠서치 최고 정확도: {0:.3f}'.format(round(grid_random_forest.best_score_ * 100, 3)))
# estimator = grid_random_forest.best_estimator_
# Y_pred = estimator.predict(X_test)

# random_forest = RandomForestClassifier(random_state=1)
# parameters = {'criterion': ['entropy', 'gini'],\
#               'max_depth': np.arange(1, 3, 1),\
#               'max_features': np.arange(1, 7, 1),\
#               'max_leaf_nodes': np.arange(5, 10, 1),\
#               'min_samples_leaf': np.arange(5, 10, 1),\
#               'min_samples_split': np.arange(5, 10, 1),\
#               'n_estimators': np.arange(5, 100, 5)}
# rs_random_forest = RandomizedSearchCV(random_forest, n_iter=100, param_distributions=parameters,\
#                                       cv=cv, random_state=1, n_jobs=-1, verbose=3)
# rs_random_forest.fit(X_train, Y_train)
# score_random_forest = pd.DataFrame(rs_random_forest.cv_results_)
# print(score_random_forest[['params', 'mean_test_score', 'rank_test_score']])
# print('랜덤 포레스트의 랜덤서치 최고 파라미터: {}'.format(rs_random_forest.best_params_))
# print('랜덤 포레스트의 랜덤서치 최고점수: {0:.3f}'.format(round(rs_random_forest.best_score_ * 100, 3)))
# estimator = rs_random_forest.best_estimator_
# Y_pred = estimator.predict(X_test)

# gbc = GradientBoostingClassifier(random_state=1)
# parameters = {'learning_rate': [(0.1) ** n for n in range(10)],\
#               'n_estimators': np.arange(10, 5000, 10),\
#               'max_depth': np.arange(1, 20, 1),\
#               'max_features': np.arange(1, 9, 1),\
#               'min_samples_leaf': np.arange(1, 10, 1),\
#               'min_samples_split': np.arange(2, 10, 1),\
#               }
# random_gbc = RandomizedSearchCV(gbc, n_iter=100, param_distributions=parameters, cv=Kfold, verbose=2, n_jobs=-1, refit=True)
# random_gbc.fit(X_train, Y_train)
# score_random_gbc = pd.DataFrame(random_gbc.cv_results_)
# print(score_random_gbc[['params', 'mean_test_score', 'rank_test_score'\
#                  ]])
# print('GBC의 랜덤서치 최고 모델: {}'.format(random_gbc.best_estimator_))
# print('GBC의 랜덤서치 최적 파라미터: {}'.format(random_gbc.best_params_))
# print('GBC의 랜덤서치 최고점수: {}'.format(round(random_gbc.best_score_ * 100, 3)))
# estimator = random_gbc.best_estimator_
# Y_pred = estimator.predict(X_test)

# mlp = MLPClassifier(max_iter=10000, hidden_layer_sizes=(100, 100), random_state=1)
# parameters = {'alpha': [(0.1) ** i for i in range(1, 5)],\
#               'activation': ['logistic', 'adam', 'lbfgs']}
# grid_mlp = GridSearchCV(mlp, param_grid=parameters, verbose=2, cv=Kfold, n_jobs=-1, refit=True)
# grid_mlp.fit(X_train, Y_train)
# score_mlp = pd.DataFrame(grid_mlp.cv_results_)
# print(score_mlp[['params', 'mean_test_score', 'rank_test_score']])
# print('MLP의 그리드서치 최적 파라미터: {}'.format(grid_mlp.best_params_))
# print('MLP의 그리드서치 최고점수: {0:.3f}'.format(round(grid_mlp.best_score_ * 100, 3)))
# estimator = grid_mlp.best_estimator_
# Y_pred = estimator.predict(X_test)

# mlp = MLPClassifier(hidden_layer_sizes=(100))
# parameters = {'alpha': [(0.1) ** n for n in range(10)],\
#                'solver': ['sgd', 'adam'],\
#                'activation': ['tanh', 'relu', 'logistic'],\
#                'max_iter': np.arange(10, 1000, 10),\
#                'batch_size': np.arange(1, 10, 1)}
# random_mlp = RandomizedSearchCV(mlp, param_distributions=parameters, random_state=1,\
#                                 n_iter=10, n_jobs=-1, cv=cv, verbose=3)
# random_mlp.fit(X_train, Y_train)
# score_random_mlp = pd.DataFrame(random_mlp.cv_results_)
# print(score_random_mlp[['params', 'mean_test_score', 'rank_test_score'\
#                         ]])
# print('MLP의 랜덤서치 최적 파라미터: {}'.format(random_mlp.best_params_))
# print('MLP의 랜덤서치 최고점수: {0:.2f}'.format(random_mlp.best_score_))
# print('MLP의 랜덤서치 최고 성능 모델: {}'.format(random_mlp.best_estimator_))
# estimator = random_mlp.best_estimator_
# Y_pred = estimator.predict(X_test)

# 결과를 CSV로 저장한다.
submission = pd.DataFrame({'PassengerId': test['PassengerId'], 'Survived': Y_pred})
# submission.to_csv('C:/python/titanic/submission.csv', index=False)