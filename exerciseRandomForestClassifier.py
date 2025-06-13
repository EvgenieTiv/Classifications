import os
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import LabelEncoder, StandardScaler, OneHotEncoder
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.preprocessing import PolynomialFeatures

# conda install pandas numpy matplotlib scikit-learn

current_dir = os.path.dirname(__file__)  # Папка, где находится .py файл
# file_path_read_train = os.path.join(сurrent_dir, "synthetic_200_rows_with_logic_filled.csv")
# file_path_read_to_predict = os.path.join(сurrent_dir, "to_predict_filled.csv")
# file_path_write = os.path.join(сurrent_dir, "to_predict_with_y.csv")

file_path_read_train = os.path.join(current_dir, "winequality_red_train.csv")
file_path_read_to_predict = os.path.join(current_dir, "winequality_red_to_predict.csv")
file_path_write = os.path.join(current_dir, "winequality_red_to_predict_with_y.csv")

# 📥 Загрузим тренировочные данные
df = pd.read_csv(file_path_read_train)

# 📌 Разделим на X (признаки) и y (целевой признак)
X = df.drop(columns=["quality"])
y = df["quality"]

# 🎯 Проверим, какие классы есть
print("📊 Уникальные классы в y:", y.unique())
print("📦 Размер X:", X.shape)
print("🎯 Размер y:", y.shape)

# 🔀 Разделим данные на train и test
# Шаг 1. Отделяем test
X_temp, X_test, y_temp, y_test = train_test_split(
    X, y, test_size=0.2, random_state=4, stratify=y   # ⬅️ сохраняет пропорции классов в train и test
)

# Шаг 2. Делим X_temp на train и validation
X_train, X_val, y_train, y_val = train_test_split(
    X_temp, y_temp, test_size=0.2, random_state=4, stratify=y_temp   # ⬅️ сохраняет пропорции классов в train и test
)

# Проверим сбалансированность классов
print("📊 Распределение классов в y_train:")
print(y_train.value_counts(normalize=True).sort_index())

print(f"Размер обучающей выборки: {X_train.shape}")
print(f"Размер валидационной выборки: {X_val.shape}")
print(f"Размер тестовой выборки: {X_test.shape}")

# 1. Определим типы признаков
categorical_features = []
numeric_features = ["fixed acidity", "volatile acidity", "citric acid", "residual sugar",
            "chlorides", "free sulfur dioxide", "total sulfur dioxide" , "density",
            "pH", "sulphates", "alcohol"]

# 2. Построим препроцессор
preprocessor = ColumnTransformer(transformers=[
    ("num", StandardScaler(), numeric_features),
    ("cat", OneHotEncoder(drop='first'), categorical_features)
])

# 3. Подготовим обучающую и тестовую выборки
X_train_prepared = preprocessor.fit_transform(X_train)
X_val_prepared = preprocessor.transform(X_val)
X_test_prepared = preprocessor.transform(X_test)  # ✅ настоящее тестовое множество

# New start
# poly = PolynomialFeatures(degree=3, include_bias=False)
# X_train_prepared = poly.fit_transform(X_train_prepared)
# X_val_prepared = poly.transform(X_val_prepared)
# X_test_prepared = poly.transform(X_test_prepared)

# После применения PolynomialFeatures
# all_feature_names = poly.get_feature_names_out()
# New end

# Grid Search с 5-кратной кросс-валидацией
# estimator	model	Это базовая модель, которую будем оптимизировать
# param_grid	param_grid	Словарь параметров и значений для перебора
# cv=5	Кросс-валидация	Делит данные на 5 частей → 4 на обучение, 1 на проверку (и так 5 раз)
# scoring='r2'	Целевая метрика	Как выбирать лучшую модель. 'r2' = коэффициент детерминации
# verbose=1	Вывод в консоль	Показывает прогресс (0 — молчать, 1 — немного, 2 — подробно)
# n_jobs=-1	Все ядра	Параллельное выполнение → быстрее

# Сетка параметров
param_grid = {
    "n_estimators": [50, 75, 100, 200],
    "max_depth": [None, 5, 10],
    "min_samples_leaf": [1, 3, 5],
    "bootstrap": [True, False]  # ⬅️ новинка
}

# n_jobs=-1	Использовать все CPU ядра	Ускоряет обучение за счёт параллелизма
model = RandomForestClassifier(random_state=4, n_jobs=-1)

grid_search = GridSearchCV(
    estimator=model,
    param_grid=param_grid,
    cv=5,
    scoring='accuracy',  # ⬅️ Метрика для классификации
    verbose=1,
    n_jobs=-1
)

# Обучение
grid_search.fit(X_train_prepared, y_train)

# Лучшие параметры и результат
# print("Модель Случайного Леса")
print("✅ Лучшие параметры:", grid_search.best_params_)
print("📈 Лучшая точность (accuracy, кросс-валидация):", grid_search.best_score_)

# Лучшая модель
best_model = grid_search.best_estimator_
y_val_pred = best_model.predict(X_val_prepared)
val_accuracy = accuracy_score(y_val, y_val_pred)

print("🎯 Лучшая модель:", grid_search.best_params_)
print(f"✅ Accuracy (cross-validation): {grid_search.best_score_:.4f}")

# Проверим на validation:
y_val_pred = best_model.predict(X_val_prepared)
val_accuracy = accuracy_score(y_val, y_val_pred)

print(f"\n🧪 Accuracy на validation set: {val_accuracy:.4f}")

# 📊 Получаем имена признаков после препроцессора
# FROM
# feature_names_num = preprocessor.named_transformers_["num"].get_feature_names_out(numeric_features)
#
# if categorical_features:
#     feature_names_cat = preprocessor.named_transformers_["cat"].get_feature_names_out(categorical_features)
# else:
#     feature_names_cat = []
#
# all_feature_names = list(feature_names_num) + list(feature_names_cat)
# TO

importances = best_model.feature_importances_
indices = sorted(range(len(importances)), key=lambda i: importances[i], reverse=True)

# print("\n🔬 Важность признаков:")
# for i in indices:
#     print(f"{all_feature_names[i]}: {importances[i]:.4f}")

# 5. Предскажем на тестовой выборке
y_pred = best_model.predict(X_test_prepared)

# 6. Оценим качество
test_accuracy = accuracy_score(y_test, y_pred)
print("\n🎯 Accuracy на тестовой выборке:", test_accuracy)
print("\n📊 Матрица ошибок:")
print(confusion_matrix(y_test, y_pred))

print("\n🧾 Отчет по классам:")
print(classification_report(y_test, y_pred, zero_division=0))

# 📥 Загружаем файл для предсказания
df_to_predict = pd.read_csv(file_path_read_to_predict)

# 🎯 Сохраняем правильные ответы
y_true = df_to_predict["quality_True"]

# 🧹 Берем только признаки (без y)
X_to_predict = df_to_predict.drop(columns=["quality_True"])

# ♻️ Преобразуем признаки (тем же препроцессором!)
X_to_predict_prepared = preprocessor.transform(X_to_predict)
# New
# X_to_predict_prepared = poly.transform(X_to_predict_prepared)

# 🔮 Предсказание
y_pred = best_model.predict(X_to_predict_prepared)

# 🎯 Оценка качества (сравниваем с y_True)
accuracy_on_to_predict = accuracy_score(y_true, y_pred)
print("📊 Accuracy на to_predict_multiclass.csv:", accuracy_on_to_predict)
print("\n📊 Матрица ошибок:")
print(confusion_matrix(y_true, y_pred))
print("\n🧾 Отчет по классам:")
print(classification_report(y_true, y_pred))

# 📝 Сохраняем предсказания в файл
df_to_predict["quality"] = y_pred
df_to_predict.to_csv(file_path_write, index=False)
print(f"\n💾 Результаты сохранены в файл: {file_path_write}")

print("\nRandom forest")
print("📊 Accuracy на разных выборках:")
print(f"Validation:  {val_accuracy:.4f}")
print(f"Test:        {test_accuracy:.4f}")
print(f"Final:        {accuracy_on_to_predict:.4f}")

# Random forest
# 📊 Accuracy на разных выборках:
# Validation:  0.6494
# Test:        0.6528
# Final:        0.6438
