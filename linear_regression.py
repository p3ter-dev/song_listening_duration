import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge 
from sklearn.metrics import mean_squared_error, r2_score

file_name = "dataset.csv"
df = pd.read_csv(file_name)

print("Dataset loaded:", df.shape)

# Keep only reasonable durations: between 20 seconds to 10 minutes (20,000 to 600,000 ms)
df = df[(df["duration_ms"] > 20000) & (df["duration_ms"] < 600000)].copy()

print("After outlier removal:", df.shape)

# create dummy binary variables for the 'track_genre' column
df_dummies = pd.get_dummies(df, columns=['track_genre'], drop_first=True)

# features from the old set, plus new ones found in the dataset
numerical_features = [
    "popularity", "danceability", "energy", "loudness", "speechiness", 
    "acousticness", "instrumentalness", "liveness", "valence", "tempo", 
    "key", "mode", "time_signature", "explicit"
]
target = "duration_ms"

# Features now include numerical ones and all the new 'track_genre_' dummy columns
genre_cols = [col for col in df_dummies.columns if col.startswith('track_genre_')]
features = numerical_features + genre_cols

X = df_dummies[features]
y = df_dummies[target] 

print(f"\nTotal number of features used: {len(features)}")

# train test and split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

scaler = StandardScaler()

# Fit scaler on training data and transform both sets
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# ridge regression
model = Ridge(alpha=1.0)
model.fit(X_train_scaled, y_train)

# evaluate
y_pred = model.predict(X_test_scaled)

mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print("\nMODEL PERFORMANCE:")
print("MSE:", f"{mse:,.2f}")
print("R² Score:", f"{r2:.4f}")
plt.figure(figsize=(9, 7))
plt.scatter(y_test, y_pred, alpha=0.3)
plt.xlabel("Actual Duration (ms)")
plt.ylabel("Predicted Duration (ms)")
plt.title("Actual vs Predicted Song Duration")
plt.grid(True)
plt.savefig("ridge_predictions.png")
plt.show()