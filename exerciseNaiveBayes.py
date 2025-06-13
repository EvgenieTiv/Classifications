import os
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import LabelEncoder, StandardScaler, OneHotEncoder
from sklearn.tree import DecisionTreeClassifier, plot_tree

# 📌 Кратко: Naive Bayes (GaussianNB)

# 🧠 Идея:
# Вычисляется вероятность, что объект принадлежит каждому классу, исходя из признаков.
# Делается независимое предположение: все признаки не зависят друг от друга (поэтому — "наивный").
# Выбирается класс с максимальной апостериорной вероятностью.

# ✅ Преимущества:
# Очень быстрый и лёгкий
# Хорошо работает, даже если предпосылки не полностью выполняются
# Часто используется как baseline

# ⚠️ Недостатки:
# Плохо справляется, если признаки сильно коррелированы
# Плохо работает с категориальными признаками (если не закодированы)

current_dir = os.path.dirname(__file__)  # Папка, где находится .py файл
# file_path_read_train = os.path.join(сurrent_dir, "synthetic_200_rows_with_logic_filled.csv")
# file_path_read_to_predict = os.path.join(сurrent_dir, "to_predict_filled.csv")
# file_path_write = os.path.join(сurrent_dir, "to_predict_with_y.csv")

file_path_read = os.path.join(current_dir, "winequality_red_train.csv")
file_path_read_to_predict = os.path.join(current_dir, "winequality_red_to_predict.csv")
file_path_write = os.path.join(current_dir, "winequality_red_to_predict_with_y.csv")

# 📥 Загрузка данных
df = pd.read_csv(file_path_read)

# 🎯 Целевая переменная и признаки
X = df.drop(columns=["quality"])
y = df["quality"]

# 🧪 Разделение на train/test
# Шаг 1. Отделяем test
X_temp, X_test, y_temp, y_test = train_test_split(
    X, y, test_size=0.2, random_state=4, stratify=y   # ⬅️ сохраняет пропорции классов в train и test
)

# Шаг 2. Делим X_temp на train и validation
X_train, X_val, y_train, y_val = train_test_split(
    X_temp, y_temp, test_size=0.2, random_state=4, stratify=y_temp   # ⬅️ сохраняет пропорции классов в train и test
)

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

# 🔗 Pipeline — это способ объединить несколько шагов (например, препроцессинг и модель) в один объект.

# 📌 GaussianNB — это модель наивного байесовского классификатора, которая:
# Предполагает, что признаки независимы
# И что каждый числовой признак в классе имеет нормальное распределение (Гаусса)
# 🔬 Что происходит внутри:
# Для каждого класса (например, "good", "poor", …) модель запоминает:
# Среднее (μ) и стандартное отклонение (σ) для каждого признака

# Когда приходит новый пример:
# Модель считает вероятность того, что он принадлежит к каждому классу, по формуле Гаусса
# Выбирает класс с максимальной вероятностью

# 📌 Это очень быстрая и простая модель, подходит даже для больших задач, если структура данных подходящая.

nb_pipeline = make_pipeline(preprocessor, GaussianNB())

# 🧠 Обучение
nb_pipeline.fit(X_train, y_train)

# 🔮 Предсказание
y_val_pred = nb_pipeline.predict(X_val)
val_accuracy = accuracy_score(y_val, y_val_pred)
print(f"\n🧪 Accuracy на validation set: {val_accuracy:.4f}")


y_test_pred = nb_pipeline.predict(X_test)


# 📊 Оценка
test_accuracy =  accuracy_score(y_test, y_test_pred)
print("\n✅ Accuracy на тестовой выборке:", test_accuracy)
print("\n📊 Матрица ошибок (test):")
print(confusion_matrix(y_test, y_test_pred))
print("\n🧾 Отчет по классам (test):")
print(classification_report(y_test, y_test_pred, zero_division=0))

# 📥 Загружаем файл для предсказания
df_to_predict = pd.read_csv(file_path_read_to_predict)

# 🎯 Сохраняем правильные ответы
y_true = df_to_predict["quality_True"]

# 🧹 Берем только признаки (без y)
X_to_predict = df_to_predict.drop(columns=["quality_True"])

# 🔮 Предсказание
y_pred = nb_pipeline.predict(X_to_predict)

# 🎯 Оценка качества (сравниваем с y_True)
accuracy_on_to_predict = accuracy_score(y_true, y_pred)
print("\n📊 Accuracy на to_predict_multiclass.csv:", accuracy_on_to_predict)
print("\n📊 Матрица ошибок:")
print(confusion_matrix(y_true, y_pred))
print("\n🧾 Отчет по классам:")
print(classification_report(y_true, y_pred, zero_division=0))

# 📝 Сохраняем предсказания в файл
df_to_predict["quality"] = y_pred
df_to_predict.to_csv(file_path_write, index=False)
print(f"\n💾 Результаты сохранены в файл: {file_path_write}")

print("\nNaive Bayes")
print("📊 Accuracy на разных выборках:")
print(f"Validation:  {val_accuracy:.4f}")
print(f"Test:        {test_accuracy:.4f}")
print(f"Final:        {accuracy_on_to_predict:.4f}")

# Naive Bayes
# 📊 Accuracy на разных выборках:
# Validation:  0.5541
# Test:        0.5521
# Final:        0.5437

