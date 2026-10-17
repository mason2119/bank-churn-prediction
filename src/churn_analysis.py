import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.tree import DecisionTreeClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import (
    accuracy_score,
    recall_score,
    precision_score,
    f1_score,
    roc_auc_score,
    classification_report,
    confusion_matrix,
    roc_curve,
    auc
)
import matplotlib.pyplot as plt
import seaborn as sns

# 设置图形风格和字体
plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams['font.family'] = 'monospace'
plt.rcParams['font.size'] = 10

# 1. 加载数据
_BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
file_path = os.path.join(_BASE, 'data', 'Customer Churn Dataset.csv')
data = pd.read_csv(file_path)

# 显示数据的前几行
print(data.head())

# 检查缺失值
print("\n缺失值统计：")
print(data.isnull().sum())

# 检查数据完整性
print("\n数据完整性检查：")
non_null_counts = data.count() / len(data)
for col, count in non_null_counts.items():
    print(f"列 {col}: {count:.2%} 非空")

# 检查数据类型
print("\n数据类型：")
print(data.dtypes)


# 3. 特征选择
# 指定特征和目标变量
X = data.drop(['RowNumber', 'CustomerId', 'Surname', 'Exited'], axis=1)
y = data['Exited']

# 确保特征列名正确
selected_features = ['CreditScore', 'Geography', 'Gender', 'Age', 'Tenure', 'Balance', 'NumOfProducts', 'HasCrCard', 'IsActiveMember', 'EstimatedSalary']
X = X[selected_features]

# 4. 处理分类变量
preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), ['CreditScore', 'Age', 'Tenure', 'Balance', 'NumOfProducts', 'EstimatedSalary']),
        ('cat', OneHotEncoder(), ['Geography', 'Gender', 'HasCrCard', 'IsActiveMember'])
    ])

# 5. 划分训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 创建包含预处理的管道
pipeline = Pipeline([
    ('preprocessor', preprocessor)
])

# 对训练集和测试集进行预处理
X_train_preprocessed = pipeline.fit_transform(X_train)
X_test_preprocessed = pipeline.transform(X_test)

# 定义一个函数来训练和评估模型
def train_and_evaluate(model, X_train, X_test, y_train, y_test, model_name):
    # 训练模型
    model.fit(X_train, y_train)
    
    # 预测
    y_pred = model.predict(X_test)
    y_pred_proba = model.predict_proba(X_test)[:, 1]
    
    # 评估指标
    acc = accuracy_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    roc_auc = roc_auc_score(y_test, y_pred_proba)
    
    # 打印结果
    print(f"{model_name} 结果：")
    print(f"准确率: {acc:.4f}")
    print(f"召回率: {recall:.4f}")
    print(f"精确率: {precision:.4f}")
    print(f"F1 分数: {f1:.4f}")
    print(f"ROC-AUC: {roc_auc:.4f}")
    print("分类报告：\n", classification_report(y_test, y_pred))
    print("混淆矩阵：\n", confusion_matrix(y_test, y_pred))
    print("\n")
    
    # 返回模型和评估指标
    return model, {
        'Accuracy': acc,
        'Recall': recall,
        'Precision': precision,
        'F1 Score': f1,
        'ROC-AUC': roc_auc
    }

# 定义一个函数来分析客户流失原因
def analyze_churn_causes(model, feature_names, model_name, top_n=5):
    # 获取特征重要性
    if hasattr(model, 'feature_importances_'):
        importances = model.feature_importances_
    else:
        print(f"{model_name} 模型无法直接提供特征重要性信息。")
        return
    
    # 将特征重要性与特征名称对应起来
    feature_importances = sorted(zip(feature_names, importances), key=lambda x: x[1], reverse=True)
    
    # 打印最重要的前 top_n 个特征
    print(f"{model_name} 客户流失原因分析（前 {top_n} 个最重要特征）：")
    for feature, importance in feature_importances[:top_n]:
        print(f"{feature}: {importance:.4f}")
    print("\n")

# 获取特征名称
numeric_features = pipeline.named_steps['preprocessor'].named_transformers_['num'].get_feature_names_out(['CreditScore', 'Age', 'Tenure', 'Balance', 'NumOfProducts', 'EstimatedSalary']).tolist()
categorical_features = pipeline.named_steps['preprocessor'].named_transformers_['cat'].get_feature_names_out(['Geography', 'Gender', 'HasCrCard', 'IsActiveMember']).tolist()
feature_names = numeric_features + categorical_features

# 6. 决策树算法
dt_model, dt_metrics = train_and_evaluate(DecisionTreeClassifier(random_state=42), X_train_preprocessed, X_test_preprocessed, y_train, y_test, "决策树")
analyze_churn_causes(dt_model, feature_names, "决策树")

# 7. 神经网络算法
nn_model, nn_metrics = train_and_evaluate(MLPClassifier(hidden_layer_sizes=(100, 50), max_iter=500, random_state=42), X_train_preprocessed, X_test_preprocessed, y_train, y_test, "神经网络")
# 神经网络模型无法直接提供特征重要性信息

# 8. 随机森林算法
rf_model, rf_metrics = train_and_evaluate(RandomForestClassifier(n_estimators=100, random_state=42), X_train_preprocessed, X_test_preprocessed, y_train, y_test, "随机森林")
analyze_churn_causes(rf_model, feature_names, "随机森林")

# 9. K 近邻算法
knn_model, knn_metrics = train_and_evaluate(KNeighborsClassifier(n_neighbors=5), X_train_preprocessed, X_test_preprocessed, y_train, y_test, "K 近邻")
# K 近邻模型无法直接提供特征重要性信息

# 比较不同算法的优劣性
metrics_df = pd.DataFrame({
    '决策树': dt_metrics,
    '神经网络': nn_metrics,
    '随机森林': rf_metrics,
    'K 近邻': knn_metrics
})

print("不同算法的性能比较：")
print(metrics_df.T)

# 定义可视化函数
def plot_model_performance(metrics_df):
    metrics = ['Accuracy', 'Recall', 'Precision', 'F1 Score', 'ROC-AUC']
    models = metrics_df.columns.tolist()

    fig, axes = plt.subplots(nrows=1, ncols=len(metrics), figsize=(20, 5))
    for i, metric in enumerate(metrics):
        axes[i].bar(models, metrics_df.loc[metric])
        axes[i].set_title(metric)
        axes[i].set_ylim([0, 1])
        for j, model in enumerate(models):
            axes[i].text(j, metrics_df.loc[metric][model] + 0.01, f"{metrics_df.loc[metric][model]:.4f}", ha='center')
    plt.tight_layout()
    plt.show()
plot_model_performance(metrics_df)


def plot_feature_importance(importances, feature_names, model_name, top_n=10):
    feature_importances = sorted(zip(feature_names, importances), key=lambda x: x[1], reverse=True)
    top_features = [f[0] for f in feature_importances[:top_n]]
    top_importances = [f[1] for f in feature_importances[:top_n]]

    plt.figure(figsize=(10, 6))
    plt.barh(top_features, top_importances)
    plt.xlabel('Importance')
    plt.title(f'{model_name} - Top {top_n} Feature Importances')
    plt.gca().invert_yaxis()
    plt.show()
plot_feature_importance(dt_model.feature_importances_, feature_names, "Decision Tree")
plot_feature_importance(rf_model.feature_importances_, feature_names, "Random Forest")



def plot_confusion_matrix(y_test, y_pred, model_name):
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['Not Exited', 'Exited'], yticklabels=['Not Exited', 'Exited'])
    plt.title(f'{model_name} Confusion Matrix')
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.show()
plot_confusion_matrix(y_test, rf_model.predict(X_test_preprocessed), "Random Forest")