import os
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler, OneHotEncoder
from sklearn.tree import DecisionTreeClassifier, plot_tree

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

current_dir = os.path.dirname(__file__)  # Папка, где находится .py файл
file_path_read_train = os.path.join(current_dir, "synthetic_200_multiclass.csv")
file_path_read_to_predict = os.path.join(current_dir, "to_predict_multiclass.csv")
file_path_write = os.path.join(current_dir, "to_predict_multiclass_with_y.csv")

# 📥 Загрузим тренировочные данные
df = pd.read_csv(file_path_read_train)

# 📌 Разделим на X (признаки) и y (целевой признак)
X = df.drop(columns=["y"])
y = df["y"]

# 🎯 Проверим, какие классы есть
print("📊 Уникальные классы в y:", y.unique())
print("📦 Размер X:", X.shape)
print("🎯 Размер y:", y.shape)

# 🔀 Разделим данные на train и test
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=4,
    stratify=y  # ⬅️ сохраняет пропорции классов в train и test
)

print("📊 Размер обучающей выборки:", X_train.shape)
print("📊 Размер тестовой выборки:", X_test.shape)

# 1. Определим типы признаков
numeric_features = ["Height", "Weight", "Age", "Score"]  # укажи свои числовые
categorical_features = ["Group"]  # укажи свои категориальные

# 2. Построим препроцессор
preprocessor = ColumnTransformer(transformers=[
    ("num", StandardScaler(), numeric_features),
    ("cat", OneHotEncoder(drop='first'), categorical_features)
])

# 3. Подготовим обучающую и тестовую выборки
X_train_prepared = preprocessor.fit_transform(X_train)
X_test_prepared = preprocessor.transform(X_test)

# 4. Обучим модель
best_depth = None
best_accuracy = 0
best_model = None

for depth in range(1, 11):
    model = DecisionTreeClassifier(max_depth=depth, random_state=4)
    model.fit(X_train_prepared, y_train)
    y_pred = model.predict(X_test_prepared)
    acc = accuracy_score(y_test, y_pred)
    print(f"max_depth={depth} ➜ accuracy={acc:.4f}")

    # выбираем модель с максимальной accuracy
    # при равной точности — берем ту, у которой глубина меньше
    if acc > best_accuracy or (acc == best_accuracy and (best_depth is None or depth < best_depth)):
        best_accuracy = acc
        best_depth = depth
        best_model = model

print(f"\n✅ Лучшая глубина дерева: {best_depth} с accuracy={best_accuracy:.4f}")

# 📊 Получаем имена признаков после препроцессора
feature_names_num = preprocessor.named_transformers_["num"].get_feature_names_out(numeric_features)
feature_names_cat = preprocessor.named_transformers_["cat"].get_feature_names_out(categorical_features)
all_feature_names = list(feature_names_num) + list(feature_names_cat)

importances = best_model.feature_importances_
indices = sorted(range(len(importances)), key=lambda i: importances[i], reverse=True)

print("\n🔬 Важность признаков:")
for i in indices:
    print(f"{all_feature_names[i]}: {importances[i]:.4f}")

# 5. Предскажем на тестовой выборке
y_pred = best_model.predict(X_test_prepared)

# 6. Оценим качество
print("\n🎯 Accuracy на тестовой выборке:", accuracy_score(y_test, y_pred))
print("\n📊 Матрица ошибок:")
print(confusion_matrix(y_test, y_pred))

print("\n🧾 Отчет по классам:")
print(classification_report(y_test, y_pred))

# 📥 Загружаем файл для предсказания
df_to_predict = pd.read_csv(file_path_read_to_predict)

# 🎯 Сохраняем правильные ответы
y_true = df_to_predict["y_True"]

# 🧹 Берем только признаки (без y)
X_to_predict = df_to_predict.drop(columns=["y", "y_True"])

# ♻️ Преобразуем признаки (тем же препроцессором!)
X_to_predict_prepared = preprocessor.transform(X_to_predict)

# 🔮 Предсказание
y_pred = best_model.predict(X_to_predict_prepared)

# 🎯 Оценка качества (сравниваем с y_True)
print("📊 Accuracy на to_predict_multiclass.csv:", accuracy_score(y_true, y_pred))
print("\n📊 Матрица ошибок:")
print(confusion_matrix(y_true, y_pred))
print("\n🧾 Отчет по классам:")
print(classification_report(y_true, y_pred))

# 📝 Сохраняем предсказания в файл
df_to_predict["y_pred"] = y_pred
df_to_predict.to_csv(file_path_write, index=False)
print(f"\n💾 Результаты сохранены в файл: {file_path_write}")

# 📉 Визуализация дерева
plt.figure(figsize=(20, 10))
plot_tree(best_model,
          feature_names=preprocessor.get_feature_names_out(),
          class_names=best_model.classes_,
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
output_path = os.path.join(current_dir, "decision_tree_decision_tree.png")
plt.savefig(output_path, dpi=300)

plt.show()