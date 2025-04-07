from imblearn.metrics import classification_report_imbalanced
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score
from sklearn.metrics import accuracy_score, roc_auc_score, classification_report
import joblib
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
import xgboost as xgb
from sklearn.neural_network import MLPClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.multiclass import OneVsRestClassifier
from sklearn.multioutput import MultiOutputClassifier,ClassifierChain
import numpy as np
import warnings
warnings.filterwarnings("ignore")
from sklearn.metrics import average_precision_score
import h5py


with h5py.File('dataset/emb_train_data_v2.h5', 'r') as hf:
    X_train = hf['X'][:]
    y_train = hf['y'][:]
    X_train_text = hf['X_text'][:]
    X_train_text_tag = hf['X_text_tag'][:]
    X_train_text = X_train_text[X_train_text_tag]
    y_train = y_train[X_train_text_tag]
    X_train = X_train[X_train_text_tag]

with h5py.File('dataset/emb_test_data_v2.h5', 'r') as hf:
    X_test = hf['X'][:]
    y_test = hf['y'][:]
    X_test_text = hf['X_text'][:]
    X_test_text_tag = hf['X_text_tag'][:]
    X_test_text = X_test_text[X_test_text_tag]
    y_test = y_test[X_test_text_tag]
    X_test = X_test[X_test_text_tag]

y_train[y_train != 1] = 0
y_test[y_test != 1] = 0
# print(y_train.shape,y_test.shape)

# mlb = MultiLabelBinarizer()
# y_train = mlb.fit_transform(y_train)
# y_test = mlb.transform(y_test)

# scaler = StandardScaler()
# X = scaler.fit_transform(X)
# X_test = scaler.transform(X_test)

model_dict = {
'nb_model':GaussianNB(),
'knn_model':KNeighborsClassifier(),
'ada_model' : AdaBoostClassifier(random_state=42),
'rf_model' : RandomForestClassifier(random_state=42, n_jobs=13),
'svc_model' : SVC(probability=True, random_state=42),
'lr_model' : LogisticRegression(max_iter=1000),
'mlp_model' : MLPClassifier(random_state=42, max_iter=1000),
'xgb_model' : xgb.XGBClassifier(objective='binary:logistic', tree_method='hist',
                        multi_strategy='multi_output_tree',random_state=42),
'gbdt_model' : GradientBoostingClassifier(random_state=42)
}

# 需封装的模型列表
ovr_models = ['nb_model','knn_model','svc_model', 'lr_model', 'ada_model', 'gbdt_model']

## Train with img emb
print(f"\n{'=' * 40}")
print('Now start to train img model')
print(f"{'=' * 40}\n")
for model_name, model in model_dict.items():
    print(f"\n{'=' * 40}")
    print(f"{model_name}")
    print(f"{'=' * 40}\n")

    if model_name in ovr_models:
        model = MultiOutputClassifier(model, n_jobs=13)

    # 使用交叉验证评估模型性能，计算 AUC
    if model_name in ovr_models or model_name=='rf_model':
        cross_val_scores = cross_val_score(
            model, X_train, y_train, cv=5,
            scoring=lambda est, X, y: roc_auc_score(
                y, 
                # 对于 MultiOutputClassifier，predict_proba 返回列表，每个元素 shape=(n_samples,2)
                np.column_stack([p[:, 1] for p in est.predict_proba(X)]), 
                average='macro'
            ),
            n_jobs=5
        )
    else:
        cross_val_scores = cross_val_score(
            model, X_train, y_train, cv=5,
            scoring=lambda est, X, y: roc_auc_score(
                y, 
                est.predict_proba(X), 
                average='macro'
            ),
            n_jobs=5
        )
    print(f"{model_name} cross_val AUC: {cross_val_scores.mean():.4f}")

    # 训练模型
    model.fit(X_train, y_train)
    joblib.dump(model, f'emb_model_weight/img_{model_name}.pkl')

    y_pred_proba = model.predict_proba(X_test)
    # print(y_pred_proba.shape,y_test.shape)
    # 计算测试集上的 AUC
    if model_name in ovr_models or model_name=='rf_model':
        test_auc = roc_auc_score(
            y_test, 
            np.column_stack([p[:, 1] for p in y_pred_proba]), 
            average='macro'
        )
        print(f"Test AUC: {test_auc:.4f}")
    else:
        test_auc = roc_auc_score(
            y_test, 
            y_pred_proba, 
            average='macro'
        )
        print(f"Test AUC: {test_auc:.4f}")

    print(classification_report(y_test, model.predict(X_test)))


print(f"\n{'=' * 40}")
print('Now start to train text model')
print(f"{'=' * 40}\n")

for model_name, model in model_dict.items():
    print(f"\n{'=' * 40}")
    print(f"{model_name}")
    print(f"{'=' * 40}\n")

    if model_name in ovr_models:
        model = MultiOutputClassifier(model, n_jobs=13)

    # 使用交叉验证评估模型性能，计算 AUC
    if model_name in ovr_models or model_name=='rf_model':
        cross_val_scores = cross_val_score(
            model, X_train_text, y_train, cv=5,
            scoring=lambda est, X, y: roc_auc_score(
                y, 
                # 对于 MultiOutputClassifier，predict_proba 返回列表，每个元素 shape=(n_samples,2)
                np.column_stack([p[:, 1] for p in est.predict_proba(X)]), 
                average='macro'
            ),
            n_jobs=5
        )
    else:
        cross_val_scores = cross_val_score(
            model, X_train_text, y_train, cv=5,
            scoring=lambda est, X, y: roc_auc_score(
                y, 
                est.predict_proba(X), 
                average='macro'
            ),
            n_jobs=5
        )
    print(f"{model_name} cross_val AUC: {cross_val_scores.mean():.4f}")

    # 训练模型
    model.fit(X_train_text, y_train)
    joblib.dump(model, f'emb_model_weight/text_{model_name}.pkl')

    y_pred_proba = model.predict_proba(X_test_text)
    # print(y_pred_proba.shape,y_test.shape)
    # 计算测试集上的 AUC
    if model_name in ovr_models or model_name=='rf_model':
        test_auc = roc_auc_score(
            y_test, 
            np.column_stack([p[:, 1] for p in y_pred_proba]), 
            average='macro'
        )
        print(f"Test AUC: {test_auc:.4f}")
    else:
        test_auc = roc_auc_score(
            y_test, 
            y_pred_proba, 
            average='macro'
        )
        print(f"Test AUC: {test_auc:.4f}")

    print(classification_report(y_test, model.predict(X_test_text)))









print(f"\n{'=' * 40}")
print('Now start to train img-text model')
print(f"{'=' * 40}\n")

X_train = np.concatenate([X_train,X_train_text],1)
X_test = np.concatenate([X_test,X_test_text],1)
for model_name, model in model_dict.items():
    print(f"\n{'=' * 40}")
    print(f"{model_name}")
    print(f"{'=' * 40}\n")

    if model_name in ovr_models:
        # model = OneVsRestClassifier(model, n_jobs=13)  # 封装为OvR
        model = MultiOutputClassifier(model, n_jobs=13)
        # model = ClassifierChain(model, n_jobs=13)

    # 使用交叉验证评估模型性能，计算 AUC
    if model_name in ovr_models or model_name=='rf_model':
        cross_val_scores = cross_val_score(
            model, X_train, y_train, cv=5,
            scoring=lambda est, X, y: roc_auc_score(
                y, 
                # 对于 MultiOutputClassifier，predict_proba 返回列表，每个元素 shape=(n_samples,2)
                np.column_stack([p[:, 1] for p in est.predict_proba(X)]), 
                average='macro'
            ),
            n_jobs=5
        )
    else:
        cross_val_scores = cross_val_score(
            model, X_train, y_train, cv=5,
            scoring=lambda est, X, y: roc_auc_score(
                y, 
                est.predict_proba(X), 
                average='macro'
            ),
            n_jobs=5
        )
    print(f"{model_name} cross_val AUC: {cross_val_scores.mean():.4f}")

    # 训练模型
    model.fit(X_train, y_train)
    joblib.dump(model, f'emb_model_weight/img_text_{model_name}.pkl')

    y_pred_proba = model.predict_proba(X_test)
    # print(y_pred_proba.shape,y_test.shape)
    # 计算测试集上的 AUC
    if model_name in ovr_models or model_name=='rf_model':
        test_auc = roc_auc_score(
            y_test, 
            np.column_stack([p[:, 1] for p in y_pred_proba]), 
            average='macro'
        )
        print(f"Test AUC: {test_auc:.4f}")
    else:
        test_auc = roc_auc_score(
            y_test, 
            y_pred_proba, 
            average='macro'
        )
        print(f"Test AUC: {test_auc:.4f}")

    print(classification_report(y_test, model.predict(X_test)))