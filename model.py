import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import joblib

# Wczytanie danych z pliku CSV
data = pd.read_csv('dane.csv')

# Podzielenie danych na cechy (X) i etykiety (y)
X = data[['Puls', 'Temperatura', 'Saturacja krwi']]
y = data['Choroba/Zaburzenie']

# Podział danych na zbiór treningowy i testowy
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = RandomForestClassifier()
model.fit(X_train, y_train)

# Przewidywanie kategorii choroby/zaburzenia dla danych testowych
y_pred = model.predict(X_test)

# Obliczenie dokładności predykcji
accuracy = accuracy_score(y_test, y_pred)
print("Dokładność modelu:", accuracy)

# Zapisanie modelu do pliku
joblib.dump(model, 'bestmodel.pkl')

#odczyt modelu
#model_x = joblib.load('bestmodel.pkl')