"""Exercise project builders and saved models without changing research outputs."""
import copy
import json
import os
from pathlib import Path
import sys
import traceback

ROOT = Path(__file__).resolve().parents[2]
OUT = Path(__file__).resolve().parent
os.environ.setdefault("MPLCONFIGDIR", "/private/tmp/tf221-matplotlib")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
os.environ.setdefault("OMP_NUM_THREADS", "2")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "2")
sys.path.insert(0, str(ROOT / "review/sequential_experiments"))
sys.path.insert(0, str(ROOT / "review/preprocessing_ablation"))

import numpy as np
import pandas as pd
import tensorflow as tf
import run_experiments as sequential
import run_ablation as ablation

assert tf.__version__ == "2.21.0", tf.__version__
frame = pd.read_csv(ROOT / "data/d_train_20180102.csv").iloc[:160].copy()
train, test = frame.iloc[:128], frame.iloc[128:]
frozen = json.loads((ROOT / "review/sequential_experiments/frozen_selection.json").read_text())
results = []


def check(name, fn):
    try:
        details = fn()
        results.append({"name": name, "ok": True, "details": details})
        print("PASS:", name, flush=True)
    except Exception:
        error = traceback.format_exc()
        results.append({"name": name, "ok": False, "error": error})
        print("FAIL:", name, error, flush=True)
    (OUT / "runtime-results.json").write_text(json.dumps(results, ensure_ascii=False, indent=2))


def classical():
    names = ["joint_xgb_089", "joint_lgb_05", "joint_hist_04", "frozen_forest", "frozen_ridge"]
    for name in names:
        spec = copy.deepcopy(frozen["specifications"][name])
        model = sequential.estimator(spec)
        model.fit(sequential.features(train, spec["variant"]), train["血糖"].to_numpy())
        pred = model.predict(sequential.features(test, spec["variant"]))
        assert len(pred) == len(test) and np.isfinite(pred).all()
    return {"models": names, "train_rows": len(train), "test_rows": len(test)}


def current_neural():
    spec = frozen["specifications"]["nn_64_nohep_raw"]
    pred, epochs, (pre, model) = sequential.fit_nn_predict(spec, train, test, epochs=2)
    assert epochs == 2 and np.isfinite(pred).all()
    path = OUT / "new-small-network.keras"
    model.save(path)
    loaded = tf.keras.models.load_model(path, compile=False)
    actual = loaded.predict(pre.transform(sequential.features(test, spec["variant"])), verbose=0).reshape(-1)
    np.testing.assert_allclose(actual, pred, atol=1e-6, rtol=1e-6)
    return {"epochs": epochs, "save_load_max_abs_diff": float(np.max(np.abs(actual - pred)))}


def original_neural():
    spec = next(s for s in ablation.definitions() if s["family"] == "nn_original")
    pre = ablation.preprocessing("scale_pca")
    x = pre.fit_transform(sequential.features(train, "all"))
    z = pre.transform(sequential.features(test, "all"))
    model = ablation.neural(x.shape[1], spec)
    hist = model.fit(x, train["血糖"].to_numpy(), epochs=2, batch_size=32, verbose=0)
    pred = model.predict(z, verbose=0)
    assert np.isfinite(hist.history["loss"]).all() and np.isfinite(pred).all()
    path = OUT / "new-original-network.keras"
    model.save(path)
    loaded = tf.keras.models.load_model(path, compile=False)
    np.testing.assert_allclose(loaded.predict(z, verbose=0), pred, atol=1e-6, rtol=1e-6)
    return {"epochs": 2, "test_rows": len(test)}


def saved_models():
    names = []
    for path in sorted((ROOT / "review").rglob("*.keras")):
        model = tf.keras.models.load_model(path, compile=False)
        dims = int(model.input_shape[-1])
        pred = model(np.zeros((3, dims), dtype="float32"), training=False).numpy()
        assert pred.shape == (3, 1) and np.isfinite(pred).all(), str(path)
        names.append(str(path.relative_to(ROOT)))
    assert len(names) == 8
    return {"models": names, "input": "synthetic zeros matching each saved input shape"}


def compare_predictions():
    before = pd.read_csv(OUT / "predictions-before.csv")
    after = pd.read_csv(OUT / "predictions-after.csv")
    assert len(after) == 1000
    np.testing.assert_array_equal(before["id"], after["id"])
    x, y = before["predicted_glucose"].to_numpy(), after["predicted_glucose"].to_numpy()
    assert np.isfinite(y).all()
    np.testing.assert_allclose(y, x, atol=1e-5, rtol=1e-6)
    return {"rows": len(after), "max_abs_diff": float(np.max(np.abs(x-y))),
            "mean_abs_diff": float(np.mean(np.abs(x-y))), "atol": 1e-5, "rtol": 1e-6}


check("project frozen classical estimators fit/predict", classical)
check("project small neural fit/predict/save/load", current_neural)
check("project original PReLU neural fit/predict/save/load", original_neural)
check("all eight historical Keras models load/infer", saved_models)
check("1000-row frozen ensemble before/after comparison", compare_predictions)
raise SystemExit(0 if all(r["ok"] for r in results) else 1)
