import os
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler, OneHotEncoder
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.model_selection import GridSearchCV

# pip install pandas numpy matplotlib scikit-learn

# 🧠 Что такое DecisionTreeClassifier
# DecisionTreeClassifier — это алгоритм машинного обучения, который используется для классификации данных,
# т.е. чтобы отнести объекты к одному из нескольких заранее известных классов
# (например, "poor", "good", "excellent").

# 📊 Идея
# Алгоритм строит дерево решений:
# На каждом шаге выбирается признак, по которому данные лучше всего делятся на классы.
# Деление продолжается до тех пор, пока:
# Лист содержит только один класс, или
# Достигнута максимальная глубина дерева / минимальное количество объектов в листе (ограничения для переобучения).

сurrent_dir = os.path.dirname(__file__)  # Папка, где находится .py файл
# file_path_read_train = os.path.join(сurrent_dir, "synthetic_200_rows_with_logic_filled.csv")
# file_path_read_to_predict = os.path.join(сurrent_dir, "to_predict_filled.csv")
# file_path_write = os.path.join(сurrent_dir, "to_predict_with_y.csv")

file_path_read_train = os.path.join(сurrent_dir, "winequality_red_train.csv")
file_path_read_to_predict = os.path.join(сurrent_dir, "winequality_red_to_predict.csv")
file_path_write = os.path.join(сurrent_dir, "winequality_red_to_predict_with_y.csv")

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
    X, y, test_size=0.2, random_state=4, stratify=y
)

# Шаг 2. Делим X_temp на train и validation
X_train, X_val, y_train, y_val = train_test_split(
    X_temp, y_temp, test_size=0.2, random_state=4, stratify=y_temp
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
X_test_prepared = preprocessor.transform(X_test)

# 4. Обучим модель
# Параметры для перебора
param_grid = {
    "max_depth": [2, 4, 6, 8, 10, 15, 20],
    "min_samples_leaf": [1, 5, 10, 20]
}

# Базовая модель
tree = DecisionTreeClassifier(random_state=4)

# GridSearchCV: автоматический подбор параметров
grid_search = GridSearchCV(tree, param_grid, cv=3, scoring="accuracy")
grid_search.fit(X_train_prepared, y_train)

# Лучшая модель
best_model = grid_search.best_estimator_
y_val_pred = best_model.predict(X_val_prepared)
val_accuracy = accuracy_score(y_val, y_val_pred)

print("🎯 Лучшая модель:", grid_search.best_params_)
print(f"✅ Accuracy (cross-validation): {grid_search.best_score_:.4f}")

# Проверим на validation:
y_val_pred = best_model.predict(X_val_prepared)
val_accuracy = accuracy_score(y_val, y_val_pred)
print(f"🧪 Accuracy на validation set: {val_accuracy:.4f}")

# 📊 Получаем имена признаков после препроцессора
feature_names_num = preprocessor.named_transformers_["num"].get_feature_names_out(numeric_features)

if categorical_features:
    feature_names_cat = preprocessor.named_transformers_["cat"].get_feature_names_out(categorical_features)
else:
    feature_names_cat = []

all_feature_names = list(feature_names_num) + list(feature_names_cat)

importances = best_model.feature_importances_
indices = sorted(range(len(importances)), key=lambda i: importances[i], reverse=True)

print("\n🔬 Важность признаков:")
for i in indices:
    print(f"{all_feature_names[i]}: {importances[i]:.4f}")

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

# 🔮 Предсказание
y_pred = best_model.predict(X_to_predict_prepared)

# 🎯 Оценка качества (сравниваем с y_True)
accuracy_on_to_predict = accuracy_score(y_true, y_pred)
print("📊 Accuracy на to_predict_multiclass.csv:", accuracy_on_to_predict)
print("\n📊 Матрица ошибок:")
print(confusion_matrix(y_true, y_pred))
print("\n🧾 Отчет по классам:")
print(classification_report(y_true, y_pred, zero_division=0))

# 📝 Сохраняем предсказания в файл
df_to_predict["quality"] = y_pred
df_to_predict.to_csv(file_path_write, index=False)
print(f"\n💾 Результаты сохранены в файл: {file_path_write}")

# 📉 Визуализация дерева
plt.figure(figsize=(20, 10))
plot_tree(best_model,
          feature_names=preprocessor.get_feature_names_out(),
          class_names=[str(c) for c in best_model.classes_],
          filled=True,
          rounded=True,
          fontsize=10)
plt.title("Decision Tree Classifier")
plt.tight_layout()

# 🔹 1. num__Score <= 0.022
# Это условие, по которому происходит разветвление.
# num__Score — означает признак Score, прошедший через StandardScaler (поэтому значения могут быть около нуля).
# <= 0.022 — это порог:
# Если Score меньше или равен 0.022, данные идут влево.
# Если больше, — вправо.
# Важно: префикс num__ взят из ColumnTransformer — это имя трансформатора "num" (числовые признаки), добавленное к имени колонки.

# 🔹 2. gini = 0.697
# Это индекс Джини — мера "неоднородности" выборки в этом узле.
# Значение 0 означает, что все объекты одного класса.
# Чем ближе к 1, тем смешаннее классы.
# Значение 0.697 означает, что в этом узле данные сильно перемешаны по классам.

# 🔹 3. samples = 160
# Количество объектов (строк) в этом узле дерева — 160 строк.

# 🔹 4. value = [54, 19, 62, 25]
# Это распределение объектов по классам:
# В этом узле:
# 54 примера класса "average"
# 19 примеров класса "excellent"
# 62 примера класса "good"
# 25 примеров класса "poor"

# Порядок соответствует class_names, которые ты передавал в plot_tree(...).
# 🔹 5. class = "good"
# Это "предсказанный класс" для узла.
# То есть если объект попадет в этот узел, модель присвоит ему
# класс "good", потому что это самый частый класс в этом узле (62 из 160).
# ✏️ Визуально:
# Цвет узла — отражает уверенность модели: чем ярче, тем однозначнее принадлежность к классу.
# Чем бледнее, тем смешаннее классы в узле.

# 💾 Сохраняем в файл
output_path = os.path.join(сurrent_dir, "decision_tree_decision_tree.png")
plt.savefig(output_path, dpi=300)

plt.show()

print("\nDecision Tree")
print("📊 Accuracy на разных выборках:")
print(f"Validation:  {val_accuracy:.4f}")
print(f"Test:        {test_accuracy:.4f}")
print(f"Final:        {accuracy_on_to_predict:.4f}")

# Decision Tree
# 📊 Accuracy на разных выборках:
# Validation:  0.5887
# Test:        0.5764
# Final:        0.5437