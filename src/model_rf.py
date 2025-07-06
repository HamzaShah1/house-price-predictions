from sklearn.compose import ColumnTransformer, make_column_selector
from sklearn.ensemble import RandomForestRegressor
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_absolute_error
from src.data import load_ames
from src.clean import clean

def train() -> None:
    df = clean(load_ames())
    X = df.drop("SalePrice", axis=1)
    y = df["SalePrice"]
    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    num_sel = make_column_selector(dtype_include="number")
    pre = ColumnTransformer([("num", "passthrough", num_sel)], remainder="drop")
    model = Pipeline(
        [
            ("pre", pre),
            ("rf", RandomForestRegressor(n_estimators=300, random_state=42)),
        ]
    )
    model.fit(X_tr, y_tr)
    preds = model.predict(X_te)
    print(f"R²: {r2_score(y_te, preds):.3f}")
    print(f"MAE: {mean_absolute_error(y_te, preds):.0f} dollars")

if __name__ == "__main__":
    train()
