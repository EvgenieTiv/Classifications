import os
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler, OneHotEncoder
from sklearn.preprocessing import PolynomialFeatures
from sklearn.model_selection import GridSearchCV

# 🧠 Что такое логистическая регрессия?
# Это модель классификации, а не регрессии, несмотря на название.
# Она предсказывает вероятность того, что объект принадлежит к определённому классу,
# чаще всего бинарному (например, y = 0 или y = 1).

# 1. Взвешивает признаки
# Каждому признаку (например, Height, Age, Group_B, ...) присваивается вес (коэффициент).
# Сначала она берёт линейную комбинацию признаков:
# z = w0 + w1*x1 + w2*x2 + ... + wn*xn

# 2. Применяет "S-образную" функцию (сигмоиду)
# Чтобы превратить результат в вероятность, логистическая регрессия применяет сигмоиду:
# p = 1 / (1 + exp(-z))

# 3. Решение — по порогу
# Обычно если p >= 0.5, модель считает, что это класс 1, иначе — класс 0.

# Пути🔧 Что делает модель во время обучения?
# # Подбирает веса w0, w1, ..., wn, чтобы максимально точно предсказывать y на обучающих данных.
# # Использует максимизацию правдоподобия — то есть ищет такие коэффициенты,
# чтобы вероятность для правильных ответов была как можно выше.

# ✅ Когда использовать логистическую регрессию:
# Когда задача — предсказать да/нет, успех/неуспех, 0 или 1.
# Когда признаки линейно связаны с вероятностью (или почти так).
# Когда важна интерпретация — логистическая регрессия показывает, какие признаки действительно важны (можно посмотреть коэффициенты).

# 🔍 Пример:
# Если Height, Age, и Group_C влияют на то, будет ли y = 1, модель это "поймёт" и даст этим признакам большие веса.

сurrent_dir = os.path.dirname(__file__)  # Папка, где находится .py файл
# file_path_read_train = os.path.join(сurrent_dir, "synthetic_200_rows_with_logic_filled.csv")
# file_path_read_to_predict = os.path.join(сurrent_dir, "to_predict_filled.csv")
# file_path_write = os.path.join(сurrent_dir, "to_predict_with_y.csv")

file_path_read_train = os.path.join(сurrent_dir, "winequality_red_train.csv")
file_path_read_to_predict = os.path.join(сurrent_dir, "winequality_red_to_predict.csv")
file_path_write = os.path.join(сurrent_dir, "winequality_red_to_predict_with_y.csv")

# Загрузка данных
df = pd.read_csv(file_path_read_train)

# Отбор признаков (X) и целевой переменной (y)
features = ["fixed acidity", "volatile acidity", "citric acid", "residual sugar",
            "chlorides", "free sulfur dioxide", "total sulfur dioxide" , "density",
            "pH", "sulphates", "alcohol"]

X = df[features]
y = df["quality"]

# Разделение на обучающую и тестовую выборки (80% / 20%)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=4)

# Проверим сбалансированность классов
print("📊 Распределение классов в y_train:")
print(y_train.value_counts(normalize=True).sort_index())

print(f"Размер обучающей выборки: {X_train.shape}")
print(f"Размер тестовой выборки: {X_test.shape}")

# Категориальные и числовые признаки
categorical_features = []
numeric_features = ["fixed acidity", "volatile acidity", "citric acid", "residual sugar",
            "chlorides", "free sulfur dioxide", "total sulfur dioxide" , "density",
            "pH", "sulphates", "alcohol"]


# StandardScaler: z = (x-mean)/std_dev
# Препроцессор: one-hot для категориальных, масштабирование для числовых
# OneHotEncoder — это способ преобразовать категориальные переменные (например, "Group": A, B, C) в набор бинарных колонок.

preprocessor = ColumnTransformer(transformers=[
    ("num", StandardScaler(), numeric_features),
    ("cat", OneHotEncoder(drop='first'), categorical_features)
])

# Преобразование признаков
X_train_prepared = preprocessor.fit_transform(X_train)
X_test_prepared = preprocessor.transform(X_test)

# Добавим полиномиальные признаки (взаимодействия и квадраты)
poly = PolynomialFeatures(degree=2, include_bias=False)
X_train_prepared = poly.fit_transform(X_train_prepared)
X_test_prepared = poly.transform(X_test_prepared)

# Обучение логистической регрессии
logreg = LogisticRegression(
    max_iter=10000
)
logreg.fit(X_train_prepared, y_train)

# Предсказания
# y_pred = logreg.predict(X_test_prepared)

# === ПРЕДСКАЗАНИЕ ДЛЯ to_predict.csv ===
# Загрузка файла
df_to_predict = pd.read_csv(file_path_read_to_predict)

# Отбор нужных колонок
X_to_predict = df_to_predict[features]  # те же признаки: Height, Weight, Age, Score, Group

# Преобразование признаков тем же препроцессором
X_to_predict_prepared = preprocessor.transform(X_to_predict)

# Преобразуем to_predict так же — через полиномиальные признаки
X_to_predict_prepared = poly.transform(X_to_predict_prepared)

# Предсказание
y_to_predict = logreg.predict(X_to_predict_prepared)

# Сравнение с y_True (если есть)
if "quality_True" in df_to_predict.columns:
    y_true = df_to_predict["quality_True"]
    accuracy_on_to_predict = accuracy_score(y_true, y_to_predict)
    print(f"🔍 Accuracy на to_predict.csv: {accuracy_on_to_predict:.2%}")
else:
    print("❗ Внимание: колонка quality_True не найдена в to_predict.csv — невозможно проверить точность.")

# Создание финального DataFrame с предсказаниями
df_predicted = df_to_predict.copy()
df_predicted["quality"] = y_to_predict

# Сохраняем результат
df_predicted.to_csv(file_path_write, index=False, encoding="utf-8-sig")
print(f"✅ Файл сохранён: {file_path_write}")
