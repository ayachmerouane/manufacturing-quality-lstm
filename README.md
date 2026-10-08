# 🏭 Prédiction de la qualité de production (LSTM)

Dashboard interactif qui prédit le **score qualité** d'une ligne de production à partir des données capteurs des machines, grâce à un **réseau de neurones LSTM** (séries temporelles).

<!-- Ajoute une capture d'écran du dashboard : -->
<!-- ![Aperçu du dashboard](docs/apercu.png) -->

## 🎯 Problématique

Anticiper une baisse de qualité avant qu'elle ne se produise, en exploitant l'historique de 4 variables machine :

| Variable | Unité |
|---|---|
| Température | °C |
| Vitesse de la machine | RPM |
| Niveau de vibration | mm/s |
| Consommation d'énergie | kWh |

**Cible :** `Production Quality Score`, mesuré chaque minute

## 🧠 Approche

- Normalisation Min-Max des variables et de la cible
- Fenêtres glissantes de **20 pas de temps** en entrée du LSTM (Keras)
- Deux modes de prédiction :
  - **Valeur unique** : prédiction du prochain score
  - **Prévision séquentielle** : prévision auto-récursive sur 1 à 20 pas
- Comparaison avec une **baseline de persistance** (RMSE sur les 100 derniers points)

## ✨ Fonctionnalités du dashboard

- 🔮 **Prédiction** : réglage des paramètres machine avec des sliders, puis prédiction et courbe historique vs prévision
- 📈 **Séries temporelles** : évolution de toutes les variables
- 📊 **Corrélations** : matrice de corrélation entre les variables et la qualité
- 📂 Chargement d'un autre fichier CSV possible

## 🛠️ Stack technique

Python · TensorFlow / Keras · scikit-learn · pandas · NumPy · Matplotlib · Streamlit

## 🚀 Lancer le projet en local

```bash
git clone https://github.com/ayachmerouane/manufacturing-quality-lstm.git
cd manufacturing-quality-lstm
pip install -r requirements.txt
streamlit run app.py
```

## 📁 Structure

```
manufacturing-quality-lstm/
├── app.py                    ← Dashboard Streamlit
├── modelDL.h5                ← Modèle LSTM entraîné
├── ManufacturingDataset.csv  ← Données capteurs (horodatées à la minute)
└── requirements.txt
```

<!-- Si tu as le RMSE du LSTM, ajoute-le ici à côté de celui de la baseline : c'est ce qui prouve que le modèle apporte quelque chose. -->

## 👤 Auteur

**Merouane Ayach** — [GitHub](https://github.com/ayachmerouane)
