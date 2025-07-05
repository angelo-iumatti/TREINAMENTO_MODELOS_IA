# Etapa 1 - Carregar e explorar os dados
# Importando bibliotecas
import pandas as pd
import numpy as np
from sklearn.datasets import load_iris

# Carregando o dataset
iris = load_iris()

# Convertendo para DataFrame
df = pd.DataFrame(data=iris.data, columns=iris.feature_names)
df['target'] = iris.target  # Adiciona a coluna de rótulos
df['species'] = pd.Categorical.from_codes(iris.target, iris.target_names)

# Visualizando os dados
df.head()

# Etapa 2 - Pré-processamento dos dados
from sklearn.model_selection import train_test_split

# Atributos (features) e rótulos (target)
X = df[iris.feature_names]
y = df['target']

# Divisão treino/teste (80% treino, 20% teste)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print("Tamanho do conjunto de treino:", X_train.shape)
print("Tamanho do conjunto de teste:", X_test.shape)

# Etapa 3 - Treinar um modelo de classificação
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier

# Modelo 1: Árvore de Decisão
dt_model = DecisionTreeClassifier(random_state=42)
dt_model.fit(X_train, y_train)

# Modelo 2: K-Nearest Neighbors (KNN)
knn_model = KNeighborsClassifier(n_neighbors=3)
knn_model.fit(X_train, y_train)

# Etapa 4 - Avaliar o desempenho do modelo
from sklearn.metrics import accuracy_score
from sklearn.model_selection import cross_val_score

# Previsões
dt_preds = dt_model.predict(X_test)
knn_preds = knn_model.predict(X_test)

# Avaliação de acurácia
print("Acurácia - Árvore de Decisão:", accuracy_score(y_test, dt_preds))
print("Acurácia - KNN:", accuracy_score(y_test, knn_preds))

# Validação cruzada
dt_cv = cross_val_score(dt_model, X, y, cv=5)
knn_cv = cross_val_score(knn_model, X, y, cv=5)

print("\nValidação Cruzada - Árvore de Decisão:", dt_cv.mean())
print("Validação Cruzada - KNN:", knn_cv.mean())

# Etapa 5 - Testar com novos exemplos
# Novas amostras: [comprimento_sépala, largura_sépala, comprimento_pétala, largura_pétala]
nova_flor = np.array([[5.1, 3.5, 1.4, 0.2]])

# Predição com ambos os modelos
print("Árvore de Decisão previu:", iris.target_names[dt_model.predict(nova_flor)[0]])
print("KNN previu:", iris.target_names[knn_model.predict(nova_flor)[0]])
