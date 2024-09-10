import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
import joblib

# Carrega o dataset
data = pd.read_csv('dataset/hand_landmarks.csv')

# Separa os atributos (landmarks) e os rótulos (letras)
X = data.iloc[:, 1:].values  # Landmarks
y = data.iloc[:, 0].values   # Labels (letras)

# Divide os dados em conjunto de treinamento e teste
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Inicializa o classificador KNN
knn = KNeighborsClassifier(n_neighbors=5)

# Treina o modelo
knn.fit(X_train, y_train)

# Faz previsões no conjunto de teste
y_pred = knn.predict(X_test)

# Avalia a precisão do modelo
accuracy = accuracy_score(y_test, y_pred)
print(f"Precisão do modelo: {accuracy * 100:.2f}%")

# Salva o modelo treinado
joblib.dump(knn, 'models/hand_model.pkl')
print("Modelo salvo em 'models/hand_model.pkl'")
