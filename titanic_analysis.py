from datetime import date
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, roc_auc_score

# Cargar datos
train_data = pd.read_csv("https://raw.githubusercontent.com/4GeeksAcademy/machine-learning-content/master/assets/titanic_train.csv")
test_data = pd.read_csv("https://raw.githubusercontent.com/4GeeksAcademy/machine-learning-content/master/assets/titanic_test.csv")
test_survived_data = pd.read_csv("https://raw.githubusercontent.com/4GeeksAcademy/machine-learning-content/master/assets/gender_submission.csv")
test_data["Survived"] = test_survived_data["Survived"]

# Combinar datos de entrenamiento y prueba
total_data = pd.concat([train_data, test_data]).reset_index(drop=True)
total_data.drop_duplicates(inplace=True)

# Eliminar columnas irrelevantes
total_data.drop(columns=["PassengerId", "Name", "Ticket", "Cabin"], inplace=True)

# Convertir variables categóricas a numéricas
total_data["Sex"] = pd.factorize(total_data["Sex"])[0]
total_data["Embarked"] = pd.factorize(total_data["Embarked"])[0]

# Rellenar valores faltantes
total_data["Age"] = total_data["Age"].fillna(total_data["Age"].median())
total_data["Embarked"] = total_data["Embarked"].fillna(total_data["Embarked"].mode()[0])
total_data["Fare"] = total_data["Fare"].fillna(total_data["Fare"].mean())

# Crear nueva característica
total_data["FamMembers"] = total_data["SibSp"] + total_data["Parch"]

# Definir variables y dividir datos
features = ["Pclass", "Age", "Fare", "Sex", "Embarked", "FamMembers"]
X = total_data[features]
y = total_data["Survived"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Normalizar los datos
scaler = StandardScaler()
X_train_norm = scaler.fit_transform(X_train)
X_test_norm = scaler.transform(X_test)

# Selección de características
selector = SelectKBest(score_func=f_classif, k=5)
X_train_sel = selector.fit_transform(X_train_norm, y_train)
X_test_sel = selector.transform(X_test_norm)
selected_features = [features[i] for i in range(len(features)) if selector.get_support()[i]]

# Convertir a DataFrame
X_train_sel = pd.DataFrame(X_train_sel, columns=selected_features)
X_test_sel = pd.DataFrame(X_test_sel, columns=selected_features)

# Entrenar modelo
model = RandomForestClassifier(random_state=42)
model.fit(X_train_sel, y_train)

# Predecir y calcular probabilidad de supervivencia
y_pred = model.predict(X_test_sel)
y_prob = model.predict_proba(X_test_sel)[:, 1]

# Evaluar el modelo
accuracy = accuracy_score(y_test, y_pred)
roc_auc = roc_auc_score(y_test, y_prob)

# Imprimir resultados
print(f"Accuracy del modelo: {accuracy:.2f}")
print(f"ROC AUC Score del modelo: {roc_auc:.2f}")
print(f"Probabilidad promedio de supervivencia: {y_prob.mean():.2f}")

# Guardar datos preprocesados
X_train_sel.to_csv("clean_titanic_train.csv", index=False)
X_test_sel.to_csv("clean_titanic_test.csv", index=False)

# Visualización del mapa de calor (descomentar para visualizar)
#plt.figure(figsize=(10, 6))
#sns.heatmap(total_data.corr(), annot=True, fmt=".2f")
#plt.tight_layout()
#plt.show()

#survival_probability = total_data["Survived"].mean()

print("Data preprocesada y guardada exitosamente.")