# ============================================================
# 学習スクリプト: 今走情報 + 前走競走結果 + 変化・負荷 - XGB (20特徴)
#  - 入力 : data_honban.csv（必須）, weather.csv（任意: 気温差を補う）
#  - 目的変数 : 異常コード (0/1)
#  - 出力 : pipeline_xgb_3group.pkl（前処理+モデルのPipeline）
#           preprocess_xgb_3group.pkl / model_xgb_3group.pkl（必要なら別保存）
#           model_meta.json（best-F1の推奨しきい値、カテゴリ語彙）
# ============================================================

import os
import sys
import json
import numpy as np
import pandas as pd
import joblib

from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score, recall_score, roc_auc_score, average_precision_score,
    precision_recall_curve
)
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.impute import SimpleImputer

# ---------------- 文字コード自動判別付き CSV ローダ ----------------
def read_csv_any(path: str) -> pd.DataFrame:
    encodings = ["utf-8-sig", "cp932", "shift_jis", "utf-16", "latin-1"]
    for enc in encodings:
        try:
            return pd.read_csv(path, encoding=enc)
        except UnicodeDecodeError:
            continue
        except Exception:
            raise
    # 最後の手段（文字化けの可能性あり）
    return pd.read_csv(path, encoding="utf-8", errors="ignore")

# ---------------- 設定 ----------------
DATA_CSV = "data_honban.csv"
WEATHER_CSV = "weather.csv"
TARGET_COL = "異常コード"

FEATURES = [
    "場所", "天気", "前走脚質", "クラス名", "年齢限定", "馬場状態",
    "競走記号", "指定条件", "トラックコード(JV)",
    "ブリンカー変化コード", "単勝オッズ", "キャリア", "前走着差タイム",
    "前走補9", "限定", "枠番", "斤量", "気温差", "前走上り3F順", "距離差"
]
CAT_COLS = ["場所", "天気", "前走脚質", "クラス名", "年齢限定", "馬場状態", "競走記号", "指定条件", "限定"]
NUM_COLS = ["トラックコード(JV)", "ブリンカー変化コード", "単勝オッズ", "キャリア",
            "前走着差タイム", "前走補9", "枠番", "斤量", "気温差", "前走上り3F順", "距離差"]

OUT_PIPE  = "pipeline_xgb_3group.pkl"
OUT_PREP  = "preprocess_xgb_3group.pkl"
OUT_MODEL = "model_xgb_3group.pkl"
OUT_META  = "model_meta.json"

# ---------------- ユーティリティ ----------------
def _to_datetime_yymmdd(val) -> pd.Timestamp:
    """yyMMdd → pandas.Timestamp"""
    s = str(val).zfill(6)
    y = int(s[:2]); m = int(s[2:4]); d = int(s[4:6])
    y += 2000 if y < 50 else 1900
    return pd.Timestamp(year=y, month=m, day=d)

def add_derived_features(df: pd.DataFrame) -> pd.DataFrame:
    """距離差 / ブリンカー変化コード / 気温差 を必要に応じて生成"""
    df = df.copy()

    # --- 距離差 ---
    if "距離差" not in df.columns and {"距離", "前距離"}.issubset(df.columns):
        df["距離差"] = pd.to_numeric(df["距離"], errors="coerce") - pd.to_numeric(df["前距離"], errors="coerce")
        print("[info] '距離差' を '距離' - '前距離' から生成しました。")

    # --- ブリンカー変化コード: code = 2*現在 + 前走 （0=0→0, 1=1→0, 2=0→1, 3=1→1） ---
    if "ブリンカー変化コード" not in df.columns and {"ブリンカー", "前走ブリンカー"}.issubset(df.columns):
        curr = pd.to_numeric(df["ブリンカー"], errors="coerce").fillna(0).astype(int).clip(0, 1)
        prev = pd.to_numeric(df["前走ブリンカー"], errors="coerce").fillna(0).astype(int).clip(0, 1)
        df["ブリンカー変化コード"] = 2 * curr + prev
        print("[info] 'ブリンカー変化コード' を 2*ブリンカー + 前走ブリンカー で生成 (0=0→0,1=1→0,2=0→1,3=1→1)。")

    # --- 気温差 ---
    if "気温差" not in df.columns:
        if {"気温", "前走気温"}.issubset(df.columns):
            df["気温差"] = pd.to_numeric(df["気温"], errors="coerce") - pd.to_numeric(df["前走気温"], errors="coerce")
            print("[info] '気温差' を '気温' - '前走気温' から生成しました。")
        elif os.path.exists(WEATHER_CSV) and {"日付", "前走日付", "場所", "前走場所"}.issubset(df.columns):
            # weather.csv をワイド→ロングに正規化
            w_raw = read_csv_any(WEATHER_CSV)
            date_col = next((c for c in w_raw.columns if c in ["日付", "date", "Date", "DATE"]), None)
            if date_col is None:
                print("[warn] weather.csv に日付列が見つかりません。'気温差' を作成できませんでした。")
                return df
            value_cols = [c for c in w_raw.columns if c != date_col]
            if len(value_cols) == 0:
                print("[warn] weather.csv の形式を解釈できません。'気温差' を作成できませんでした。")
                return df
            w_long = w_raw.melt(id_vars=[date_col], var_name="場所", value_name="気温").copy()
            w_long.rename(columns={date_col: "w_date"}, inplace=True)
            w_long["w_date"] = pd.to_datetime(w_long["w_date"], errors="coerce")

            # 今走/前走の日時
            now_dt  = df["日付"].apply(_to_datetime_yymmdd)
            prev_dt = df["前走日付"].apply(_to_datetime_yymmdd)

            # 今走 merge
            m_now = df.assign(_now_dt=now_dt).merge(
                w_long, left_on=["_now_dt", "場所"], right_on=["w_date", "場所"], how="left"
            ).rename(columns={"気温": "今走気温"})
            # 前走 merge（前走場所で結合）
            m_prev = m_now.assign(_prev_dt=prev_dt).merge(
                w_long, left_on=["_prev_dt", "前走場所"], right_on=["w_date", "場所"], how="left",
                suffixes=("", "_prev")
            )
            prev_temp_col = "気温_prev" if "気温_prev" in m_prev.columns else "気温"
            m_prev = m_prev.rename(columns={prev_temp_col: "前走気温"})
            m_prev["気温差"] = pd.to_numeric(m_prev["今走気温"], errors="coerce") - pd.to_numeric(m_prev["前走気温"], errors="coerce")
            df = m_prev.drop(columns=["w_date", "_now_dt", "_prev_dt"], errors="ignore")
            print("[info] weather.csv(ワイド表)から '気温差' を生成（今走[日付×場所] - 前走[日付×前走場所]）。")
        else:
            print("[warn] '気温差' を作成するための列（日付/前走日付/場所/前走場所）または weather.csv が不足しています。")

    return df

# ------------------- main -------------------
def main():
    # --- データ読込 ---
    df = read_csv_any(DATA_CSV)
    if TARGET_COL not in df.columns:
        print(f"[error] 目的変数 '{TARGET_COL}' が見つかりません。")
        sys.exit(1)

    # --- 派生特徴 ---
    df = add_derived_features(df)

    # --- 必須列チェック ---
    missing = [c for c in FEATURES if c not in df.columns]
    if missing:
        print("[error] 学習に必要な列が不足:", missing)
        print("CSV か 列名 を確認してください。")
        sys.exit(1)

    # --- 目的変数整形（NaNは除外） ---
    y = pd.to_numeric(df[TARGET_COL], errors="coerce").astype("Int64")
    mask = y.notna()
    df = df.loc[mask].copy()
    y = y.loc[mask].astype(int)

    # --- 欠損/型の前処理 ---
    # 1) カテゴリ列: 空文字や空白のみ → NaN → "なし" を正式カテゴリとして扱う
    df[CAT_COLS] = df[CAT_COLS].apply(lambda s: s.replace(r"^\s*$", np.nan, regex=True))
    cat_df = df[CAT_COLS].fillna("なし").astype(str)
    categories_list = [sorted(set(cat_df[c].tolist() + ["なし"])) for c in CAT_COLS]

    # 2) 数値列: 確実に数値化（失敗は NaN → 中央値で補完）
    for c in NUM_COLS:
        df[c] = pd.to_numeric(df[c], errors="coerce")

    # --- 前処理パイプライン ---
    pre = ColumnTransformer([
        ("cat", Pipeline([
            ("imp", SimpleImputer(strategy="constant", fill_value="なし")),
            ("oh", OneHotEncoder(
                categories=categories_list,  # "なし" を必ず含める
                handle_unknown="ignore",
                sparse_output=True
            )),
        ]), CAT_COLS),
        ("num", Pipeline([
            ("imp", SimpleImputer(strategy="median")),
        ]), NUM_COLS)
    ])

    # --- クラス重み（目安）---
    pos = y.sum()
    neg = len(y) - pos
    spw = max((neg / max(pos, 1)), 1.0)

    model = XGBClassifier(
        objective="binary:logistic",
        eval_metric="logloss",
        n_estimators=800,
        learning_rate=0.05,
        max_depth=6,
        subsample=0.8,
        colsample_bytree=0.8,
        reg_lambda=1.0,
        random_state=42,
        scale_pos_weight=spw,
        n_jobs=0,
    )

    pipe = Pipeline([("prep", pre), ("model", model)])

    # --- 学習 ---
    X = df[FEATURES].copy()
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    pipe.fit(X_train, y_train)

    # ===== 監査：データ分布と確率分布 =====
    print("[audit] rows =", len(df))
    print(f"[audit] label counts: pos={int(y.sum())} neg={len(y)-int(y.sum())}  pos_rate={y.mean():.3f}")
    print(f"[audit] scale_pos_weight used = {spw:.2f}")

    for split_name, Xs, ys in [("train", X_train, y_train), ("test", X_test, y_test)]:
        prob = pipe.predict_proba(Xs)[:, 1]
        qs = np.quantile(prob, [0, 0.25, 0.5, 0.75, 0.9, 1.0])
        print(f"[audit] {split_name} proba quantiles:", [f"{q:.3f}" for q in qs])
        print(f"[audit] {split_name} fraction>0.9:", f"{np.mean(prob>0.9):.3f}")

    # --- best-F1 しきい値 ---
    prob_test = pipe.predict_proba(X_test)[:, 1]
    prec, rec, thr = precision_recall_curve(y_test, prob_test)
    f1 = (2*prec[:-1]*rec[:-1]) / (prec[:-1] + rec[:-1] + 1e-9)
    best_idx = int(np.argmax(f1))
    best_thr = float(thr[best_idx])
    print(f"[audit] best-F1 threshold = {best_thr:.3f}  (F1={f1[best_idx]:.3f})")

    # --- 簡易評価（デフォ 0.5 と best-F1 の2本） ---
    for name, thruse in [("0.50", 0.5), (f"{best_thr:.3f}", best_thr)]:
        proba = pipe.predict_proba(X_test)[:, 1]
        pred = (proba >= thruse).astype(int)
        acc = accuracy_score(y_test, pred)
        rcl = recall_score(y_test, pred)
        roc = roc_auc_score(y_test, proba)
        prc = average_precision_score(y_test, proba)
        print(f"[score thr={name}] Accuracy={acc:.3f}  Recall={rcl:.3f}  ROC_AUC={roc:.3f}  PR_AUC={prc:.3f}")

    # --- 保存 ---
    joblib.dump(pipe, OUT_PIPE);    print(f"[save] Pipeline → {OUT_PIPE}")
    joblib.dump(pre, OUT_PREP);     print(f"[save] Preprocess → {OUT_PREP}")
    joblib.dump(model, OUT_MODEL);  print(f"[save] Model → {OUT_MODEL}")

    # 推奨しきい値・カテゴリ語彙の保存（アプリ側で既定値やプルダウン同期に使用）
    cat_vocab = {c: categories_list[i] for i, c in enumerate(CAT_COLS)}
    with open(OUT_META, "w", encoding="utf-8") as f:
        json.dump({"best_f1_threshold": best_thr, "cat_vocab": cat_vocab}, f, ensure_ascii=False, indent=2)
    print(f"[save] Meta → {OUT_META}")

    # --- 重要度（One-Hot後を元列で集約） ---
    try:
        oh = pipe.named_steps["prep"].named_transformers_["cat"].named_steps["oh"]
        cat_names = oh.get_feature_names_out(CAT_COLS)
        num_names = np.array(NUM_COLS)
        feat_names = np.concatenate([cat_names, num_names])
        importances = pipe.named_steps["model"].feature_importances_
        imp_df = pd.DataFrame({"feature": feat_names, "importance": importances})
        imp_df["base"] = imp_df["feature"].str.split("__").str[0]
        agg = imp_df.groupby("base", as_index=False)["importance"].sum().sort_values("importance", ascending=False)
        print("[top20 by original columns]\n", agg.head(20))
    except Exception as e:
        print("[warn] 重要度の集計に失敗:", e)

if __name__ == "__main__":
    main()
