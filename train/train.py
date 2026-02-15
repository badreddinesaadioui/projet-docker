"""
Train the Engie DNN: load data from /data, preprocess, run Keras Tuner,
train the best model, save model + scaler + feature list to /model.
"""
import os
import json
import time
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import joblib
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, callbacks
import keras_tuner as kt

DATA_DIR = os.environ.get("DATA_DIR", "/data")
MODEL_DIR = os.environ.get("MODEL_DIR", "/model")
BATCH_SIZE = int(os.environ.get("BATCH_SIZE", "512"))
EPOCHS = int(os.environ.get("EPOCHS", "200"))
MAX_TRIALS = int(os.environ.get("MAX_TRIALS", "2"))


def load_and_preprocess():
    """Load the two CSVs, merge them, fill missing values per turbine, one-hot encode
    MAC_CODE, split in time (70% train, 15% val, 15% test), then scale features."""
    df_x = pd.read_csv(os.path.join(DATA_DIR, "engie_X.csv"), sep=";")
    df_y = pd.read_csv(os.path.join(DATA_DIR, "engie_Y.csv"), sep=";")
    df = df_x.merge(df_y, on="ID", how="inner")

    data = df.copy()
    numeric_cols_to_impute = [
        c for c in data.select_dtypes(include=[np.number]).columns
        if c not in ["ID", "Date_time", "TARGET"]
    ]
    for col in numeric_cols_to_impute:
        if data[col].isnull().any():
            data[col] = data.groupby("MAC_CODE")[col].transform(
                lambda x: x.fillna(x.median())
            )

    data = pd.get_dummies(data, columns=["MAC_CODE"], prefix="turbine", dtype=int)
    data = data.sort_values("Date_time").reset_index(drop=True)

    cols_to_exclude = ["ID", "Date_time", "TARGET"]
    feature_cols = [c for c in data.columns if c not in cols_to_exclude]

    n = len(data)
    train_end = int(n * 0.7)
    val_end = int(n * 0.85)
    train_data = data.iloc[:train_end]
    val_data = data.iloc[train_end:val_end]
    test_data = data.iloc[val_end:]

    X_train = train_data[feature_cols].values
    y_train = train_data["TARGET"].values
    X_val = val_data[feature_cols].values
    y_val = val_data["TARGET"].values
    X_test = test_data[feature_cols].values
    y_test = test_data["TARGET"].values

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)
    X_test_scaled = scaler.transform(X_test)

    return (
        X_train_scaled, y_train,
        X_val_scaled, y_val,
        X_test_scaled, y_test,
        scaler,
        feature_cols,
    )


def build_tunable_model(hp, input_dim):
    """Build a DNN whose architecture and learning rate are chosen by Keras Tuner."""
    model = keras.Sequential()
    model.add(layers.Input(shape=(input_dim,)))
    n_layers = hp.Int("n_layers", 2, 5)
    for i in range(n_layers):
        units = hp.Int(f"units_{i}", 32, 512, step=32)
        model.add(layers.Dense(units, activation="relu"))
        if hp.Boolean(f"batch_norm_{i}"):
            model.add(layers.BatchNormalization())
        dropout_rate = hp.Float(f"dropout_{i}", 0.0, 0.5, step=0.1)
        if dropout_rate > 0:
            model.add(layers.Dropout(dropout_rate))
    model.add(layers.Dense(1))
    lr = hp.Choice("learning_rate", [1e-2, 5e-3, 1e-3, 5e-4, 1e-4])
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=lr),
        loss="mse",
        metrics=["mae"],
    )
    return model


def main():
    t_start = time.time()
    print("Loading and preprocessing data from", DATA_DIR, "...")
    (
        X_train, y_train,
        X_val, y_val,
        X_test, y_test,
        scaler,
        feature_cols,
    ) = load_and_preprocess()
    input_dim = X_train.shape[1]
    print("Train:", X_train.shape[0], "Val:", X_val.shape[0], "Test:", X_test.shape[0], "Features:", input_dim)

    def _builder(hp):
        return build_tunable_model(hp, input_dim)

    print("Running Keras Tuner (max_trials =", MAX_TRIALS, ") ...")
    tuner = kt.BayesianOptimization(
        _builder,
        objective=kt.Objective("val_mae", direction="min"),
        max_trials=MAX_TRIALS,
        num_initial_points=min(2, MAX_TRIALS),
        directory=os.path.join(MODEL_DIR, "tuner_dir"),
        project_name="engie_dnn",
        overwrite=True,
    )
    tuner.search(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=50,
        batch_size=BATCH_SIZE,
    )

    best_hps = tuner.get_best_hyperparameters(num_trials=1)[0]
    best_model = tuner.hypermodel.build(best_hps)
    early_stop = callbacks.EarlyStopping(
        monitor="val_mae", patience=10, restore_best_weights=True
    )
    print("Training the best model (epochs =", EPOCHS, ", batch_size =", BATCH_SIZE, ") ...")
    best_model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        callbacks=[early_stop],
        verbose=1,
    )

    y_pred_test = best_model.predict(X_test, batch_size=BATCH_SIZE).flatten()
    y_pred_train = best_model.predict(X_train, batch_size=BATCH_SIZE).flatten()
    y_pred_val = best_model.predict(X_val, batch_size=BATCH_SIZE).flatten()

    mae_test = mean_absolute_error(y_test, y_pred_test)
    rmse_test = np.sqrt(mean_squared_error(y_test, y_pred_test))
    r2_test = r2_score(y_test, y_pred_test)
    mae_train = mean_absolute_error(y_train, y_pred_train)
    r2_train = r2_score(y_train, y_pred_train)
    mae_val = mean_absolute_error(y_val, y_pred_val)
    r2_val = r2_score(y_val, y_pred_val)

    elapsed = time.time() - t_start
    print("Test  — MAE:", round(mae_test, 4), "RMSE:", round(rmse_test, 4), "R2:", round(r2_test, 4))
    print("Train — MAE:", round(mae_train, 4), "R2:", round(r2_train, 4))
    print("Val   — MAE:", round(mae_val, 4), "R2:", round(r2_val, 4))
    print("Total time:", round(elapsed, 1), "s")

    os.makedirs(MODEL_DIR, exist_ok=True)
    model_path = os.path.join(MODEL_DIR, "dnn_model.keras")
    scaler_path = os.path.join(MODEL_DIR, "scaler.pkl")
    features_path = os.path.join(MODEL_DIR, "feature_columns.json")
    best_model.save(model_path)
    joblib.dump(scaler, scaler_path)
    with open(features_path, "w") as f:
        json.dump(feature_cols, f, indent=2)

    metrics = {
        "train": {"MAE": round(mae_train, 4), "R2": round(r2_train, 4)},
        "validation": {"MAE": round(mae_val, 4), "R2": round(r2_val, 4)},
        "test": {"MAE": round(mae_test, 4), "RMSE": round(rmse_test, 4), "R2": round(r2_test, 4)},
        "total_time_seconds": round(elapsed, 1),
    }
    metrics_path = os.path.join(MODEL_DIR, "training_metrics.json")
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2)
    print("Done. Model, scaler, feature list and training_metrics.json saved in", MODEL_DIR)


if __name__ == "__main__":
    main()
