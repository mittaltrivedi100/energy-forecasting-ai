from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
import joblib

def train_model(df):
    X = df[['temperature', 'humidity', 'production_rate', 'hour', 'dayofweek']]
    y = df['energy_consumption']
    X_train, _, y_train, _ = train_test_split(X, y, test_size=0.2, random_state=42)

    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    joblib.dump(model, 'energy_model.pkl')
    return model
