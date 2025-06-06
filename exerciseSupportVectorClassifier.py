import os
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.svm import SVC

# 🧠 Что такое SVC?
# 📌 Кратко:
# SVC ищет границу (гиперплоскость), которая максимально разделяет классы с наибольшим отступом.
# Это очень мощный метод, особенно:
# когда классы не линейно разделимы
# при использовании ядер (kernels)

# Важно:
# Обязательно масштабирование признаков!
# Медленно работает на больших выборках, но у нас пока всё нормально

сurrent_dir = os.path.dirname(__file__)  # Папка, где находится .py файл
# file_path_read_train = os.path.join(сurrent_dir, "synthetic_200_rows_with_logic_filled.csv")
# file_path_read_to_predict = os.path.join(сurrent_dir, "to_predict_filled.csv")
# file_path_write = os.path.join(сurrent_dir, "to_predict_with_y.csv")

file_path_read = os.path.join(сurrent_dir, "winequality_red_train.csv")
file_path_read_to_predict = os.path.join(сurrent_dir, "winequality_red_to_predict.csv")
file_path_write = os.path.join(сurrent_dir, "winequality_red_to_predict_with_y.csv")

# 📥 Загрузка данных
df = pd.read_csv(file_path_read)

# 🎯 Целевая переменная и признаки
X = df.drop(columns=["quality"])
y = df["quality"]

# 🧪 Разделение на train/test
X_train, X_test, y_train, y_test = train_test_split(X,
                                                    y,
                                                    test_size=0.25,
                                                    random_state=4,
                                                    stratify=y)   # ⬅️ сохраняет пропорции классов в train и test

# 🔍 Найдём категориальные и числовые признаки
# GaussianNB по-прежнему предполагает, что признаки (в том числе бинарные) имеют нормальное распределение, что не совсем верно для one-hot признаков.
# Это значит, что модель:
# всё ещё делает сильные предположения
# может работать хуже, если:
# признаки не независимы (например, Height и Weight)
# распределение сильно негауссовское

categorical_cols = X.select_dtypes(include=["object", "category"]).columns.tolist()
numerical_cols = X.select_dtypes(include=["int64", "float64"]).columns.tolist()

# ⚙️ Создаём трансформер
preprocessor = ColumnTransformer(
    transformers=[
        ("num", StandardScaler(), numerical_cols),
        ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_cols)
    ]
)

# 🔧 Объединяем препроцессор и классификатор
# 📌 kernel='rbf'
# Что это: способ, которым SVC "видит" расстояние между точками.

# Значение	Что делает	Подходит когда
# 'linear'	Простая линейная граница	Если классы линейно разделимы
# 'rbf'	Радиальная функция (Гауссовское ядро)	✅ Подходит почти всегда
# 'poly'	Полиномиальное ядро	Для сложных, но регулярных границ
# 'sigmoid'	Имитация нейросети (редко используется)

# 📌 C=1.0
# Что это: баланс между ошибками на обучении и размером границы между классами.
# Значение	Что делает
# Меньше (0.1, 0.01)	Больше допускает ошибок, но граница проще — менее переобучается
# Больше (10, 100)	Меньше ошибок, но граница может стать сложной — больше риск переобучения

# 📌 gamma='scale'
# Что это: как далеко распространяется влияние каждой точки (важно для rbf, poly).

# Значение	Что делает
# 'scale' (по умолчанию)	1 / (n_features × Var(X))
# 'auto'	1 / n_features
# float (например, 0.1, 10)	вручную — чем меньше, тем "шире" гаусс

# 📌 Если gamma слишком большое → модель может переобучиться.
# Если слишком маленькое → будет "плоская", не чувствительная.
# 🟩 Рекомендуется использовать 'scale' на старте и подбирать потом.

# 📌 probability=True
# Что это: позволяет использовать predict_proba()
# (иначе модель даёт только predict() — жёсткие метки, без вероятностей)

# ⚠️ Замедляет обучение
# ✅ Но нужен, если ты планируешь использовать вероятности (например, для уверенности или ROC-кривых)

# Если ты не используешь predict_proba() — можешь убрать этот параметр.

param_grid = {
    'svc__C': [0.1, 1, 10],
    'svc__gamma': ['scale', 0.1, 1],
    'svc__kernel': ['rbf', 'linear']
}

grid = GridSearchCV(make_pipeline(preprocessor, SVC()), param_grid, cv=5)
grid.fit(X_train, y_train)

print("Лучшие параметры:", grid.best_params_)

# # 🧠 Обучение
grid.fit(X_train, y_train)

# # 🔮 Предсказание
y_pred = grid.predict(X_test)

# # 📊 Оценка
print("🎯 Accuracy на тестовой выборке:", accuracy_score(y_test, y_pred))
print("\n📊 Матрица ошибок:")
print(confusion_matrix(y_test, y_pred))
print("\n🧾 Отчет по классам:")
print(classification_report(y_test, y_pred))

# 📥 Предсказание на реальных данных
df_to_predict = pd.read_csv(file_path_read_to_predict)
X_to_predict = df_to_predict.drop(columns=["quality_True"])
y_true = df_to_predict["quality_True"]

# 🔮 Предсказание
y_pred_real = grid.predict(X_to_predict)

# 📊 Оценка
print("\n📊 Accuracy на to_predict_multiclass.csv:", accuracy_score(y_true, y_pred_real))
print("\n📊 Матрица ошибок:")
print(confusion_matrix(y_true, y_pred_real))
print("\n🧾 Отчет по классам:")
print(classification_report(y_true, y_pred_real))

# 💾 Сохраняем
df_to_predict["quality"] = y_pred_real
df_to_predict.to_csv(file_path_write, index=False)
print(f"\n💾 Результаты сохранены в файл: {file_path_write}")