import os
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import LabelEncoder, StandardScaler, OneHotEncoder
from sklearn.tree import DecisionTreeClassifier, plot_tree

# 🧭 K-Nearest Neighbors Classifier (KNN)
# 📌 Что это:
# KNeighborsClassifier — это один из самых простых и понятных алгоритмов классификации.
# Он не строит модель явно, а просто сохраняет все обучающие данные и:

# 1. Когда приходит новый объект, KNN ищет K ближайших соседей среди обучающих данных.
# 2. Находит класс, который встречается чаще всего среди этих K.
# 3. Присваивает этот класс объекту.

# 🎓 Преимущества:

# Очень прост и интуитивно понятен.
# Часто даёт хорошие результаты на небольших и чистых датасетах.

# ⚠️ Недостатки:

# Медленно работает на больших датасетах, потому что сравнивает с каждым примером.
# Чувствителен к масштабу признаков — поэтому обязательно масштабирование (например, StandardScaler).
# Зависит от выбора k.

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
# Шаг 1. Отделяем test
X_temp, X_test, y_temp, y_test = train_test_split(
    X, y, test_size=0.2, random_state=4, stratify=y   # ⬅️ сохраняет пропорции классов в train и test
)

# Шаг 2. Делим X_temp на train и validation
X_train, X_val, y_train, y_val = train_test_split(
    X_temp, y_temp, test_size=0.2, random_state=4, stratify=y_temp   # ⬅️ сохраняет пропорции классов в train и test
)

# 🔍 Найдём категориальные и числовые признаки
categorical_cols = X.select_dtypes(include=["object", "category"]).columns.tolist()
numerical_cols = X.select_dtypes(include=["int64", "float64"]).columns.tolist()

# ⚙️ Создаём трансформер
preprocessor = ColumnTransformer(
    transformers=[
        ("num", StandardScaler(), numerical_cols),
        ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_cols)
    ]
)

# 🏗️ Преобразуем данные
X_train_scaled = preprocessor.fit_transform(X_train)
X_val_scaled = preprocessor.transform(X_val)
X_test_scaled = preprocessor.transform(X_test)  # ✅ настоящее тестовое множество

# 🧠 Обучение KNN
param_grid = {'n_neighbors': list(range(1, 21))}

grid_search = GridSearchCV(KNeighborsClassifier(), param_grid, cv=5, scoring='accuracy')

grid_search.fit(X_train_scaled, y_train)

print("🔍 Лучшее значение k:", grid_search.best_params_)
print("📈 Лучшая средняя точность (cv):", grid_search.best_score_)

# 🔁 Обучим модель с лучшим k
knn = grid_search.best_estimator_
y_val_pred = knn.predict(X_val_scaled)
val_accuracy = accuracy_score(y_val, y_val_pred)
print(f"\n🧪 Accuracy на validation set: {val_accuracy:.4f}")

# 5. Предскажем на тестовой выборке
y_pred = knn.predict(X_test_scaled)

# 📊 Оценка качества
test_accuracy = accuracy_score(y_test, y_pred)
print("\n🎯 Accuracy на тестовой выборке:", test_accuracy)
print("\n📋 Classification Report:\n", classification_report(y_test, y_pred))
print("📉 Confusion Matrix:\n", confusion_matrix(y_test, y_pred))

# 📥 Загружаем файл для предсказания
df_to_predict = pd.read_csv(file_path_read_to_predict)

# 🎯 Сохраняем правильные ответы
y_true = df_to_predict["quality_True"]

# 🧹 Берем только признаки (без y)
X_to_predict = df_to_predict.drop(columns=["quality_True"])

# ♻️ Преобразуем признаки (тем же препроцессором!)
X_to_predict_prepared = preprocessor.transform(X_to_predict)


# 🎯 Оценка качества (сравниваем с y_True)
y_pred_predict = knn.predict(X_to_predict_prepared)
accuracy_on_to_predict = accuracy_score(y_true, y_pred_predict)
print("\n📊 Accuracy на to_predict_multiclass.csv:", accuracy_on_to_predict)
print("\n📊 Матрица ошибок:")
print(confusion_matrix(y_true, y_pred_predict))
print("\n🧾 Отчет по классам:")
print(classification_report(y_true, y_pred_predict, zero_division=0))

# 📝 Сохраняем предсказания в файл
df_to_predict["quality"] = y_pred_predict
df_to_predict.to_csv(file_path_write, index=False)
print(f"\n💾 Результаты сохранены в файл: {file_path_write}")

print("\nKNN Classifier")
print("📊 Accuracy на разных выборках:")
print(f"Validation:  {val_accuracy:.4f}")
print(f"Test:        {test_accuracy:.4f}")
print(f"Final:        {accuracy_on_to_predict:.4f}")

# KNN Classifier
# 📊 Accuracy на разных выборках:
# Validation:  0.6190
# Test:        0.5521
# Final:        0.5875
