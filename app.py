# app.py --- 推論専用（学習は別スクリプトで作成した Pipeline を使用）
import os, json, pickle
import numpy as np
import pandas as pd
import streamlit as st

# joblib が無い環境でも動くようにフォールバック
try:
    import joblib
except Exception:
    joblib = None

st.set_page_config(page_title="Injury Predictor", page_icon="🐎", layout="wide")
st.title("競走馬の怪我予測")

st.markdown(
    """

<p style="color: #6c757d; font-size: 0.95rem;">
競走馬のレース中の怪我を予測するWebアプリケーションです。<br>
「怪我の可能性が高いか／無事完走する可能性が高いか」を推定します。<br>
※実際のレースでは怪我をしない馬が大半を占めるため、実データと同じ割合で学習を行うと、すべての馬を「怪我なし」と予測した場合に精度が最大になってしまうおそれがあります。そのため本モデルでは、学習データにおいて「怪我あり」と「怪我なし」の件数を半々に調整し、怪我をする馬のパターンがより浮かび上がるようにしています。<br>
※精度は約62％です。<br>
※本アプリケーションは獣医師・厩舎関係者の判断を補助するものであり、専門家の判断に代わるものではありません。
</p>

""",
    unsafe_allow_html=True,
)

# === 学習で使った20特徴（列名は厳密一致） ===
FEATURES = [
    "場所","天気","前走脚質","クラス名","年齢限定","馬場状態","競走記号","指定条件","トラックコード(JV)",
    "ブリンカー変化コード","単勝オッズ","キャリア","前走着差タイム","前走補9","限定","枠番","斤量","気温差","前走上り3F順","距離差"
]

# === ブリンカー：内部値は 0/1/2/3、表示は人間語 ===
SHOW_CODES = False  # True にすると "0: 未装着→未装着" のようにコードも併記
BLINKER_OPTIONS = [0, 1, 2, 3]
BLINKER_LABELS = {
    0: "未装着→未装着（継続）",
    1: "装着→未装着（外す）",
    2: "未装着→装着（付ける）",
    3: "装着→装着（継続）",
}
def _fmt_blinker(v: int) -> str:
    return f"{v}: {BLINKER_LABELS[v]}" if SHOW_CODES else BLINKER_LABELS[v]

# === JVトラックコード：UI では数値を見せず、ラベルだけ ===
TRACK_CODE_LABELS = {
    10: "芝・直",
    11: "芝・左",
    12: "芝・左 外回り",
    13: "芝・左 内-外",
    14: "芝・左 外-内",
    15: "芝・左 内2周",
    16: "芝・左 外2周",
    17: "芝・右",
    18: "芝・右 外回り",
    19: "芝・右 内-外",
    20: "芝・右 外-内",
    21: "芝・右 内2周",
    22: "芝・右 外2周",
    23: "ダート・左",
    24: "ダート・右",
    25: "ダート・左 内回り",
    26: "ダート・右 外回り",
    27: "サンド・左",
    28: "サンド・右",
    29: "ダート・直",
}
TRACK_CODE_KEYS = list(TRACK_CODE_LABELS.keys())

# === 既定ファイル名 ===
DEFAULT_PIPE_CANDIDATES = ["pipeline_xgb_3group.pkl","pipeline.pkl","pipeline.joblib"]
DEFAULT_META = "model_meta.json"  # {"best_f1_threshold": 0.42} など

# ---------- ローダ ----------
def _load_pipeline(uploaded_file=None):
    if uploaded_file is not None:
        if joblib is not None:
            try:
                return joblib.load(uploaded_file), f"uploaded:{uploaded_file.name}"
            except Exception:
                pass
        if hasattr(uploaded_file, "seek"):
            uploaded_file.seek(0)
        return pickle.load(uploaded_file), f"uploaded:{getattr(uploaded_file,'name','file')}"
    # デフォルト（フォルダ内）から探す
    for p in DEFAULT_PIPE_CANDIDATES:
        if os.path.exists(p):
            if joblib is not None:
                try:
                    return joblib.load(p), p
                except Exception:
                    pass
            with open(p, "rb") as f:
                return pickle.load(f), p
    return None, None

def _load_meta(uploaded_meta=None):
    if uploaded_meta is not None:
        try:
            return json.load(uploaded_meta), f"uploaded:{uploaded_meta.name}"
        except Exception:
            return None, None
    if os.path.exists(DEFAULT_META):
        with open(DEFAULT_META, "r", encoding="utf-8") as f:
            return json.load(f), DEFAULT_META
    return None, None

# ---------- 学習パイプラインから選択肢を復元 ----------
def _choices_from_pipeline_exact(pipeline, meta):
    """OneHotEncoder の categories_ をそのまま UI 候補に使う（表記は学習時のまま）。"""
    FALLBACK = {
        "場所": ["札幌","函館","福島","新潟","東京","中山","中京","京都","阪神","小倉"],
        "天気": ["晴","曇","雨","雪","小雨","小雪","その他"],
        "前走脚質": ["逃げ","先行","中団","ﾏｸﾘ","後方"],
        "クラス名": ["新馬","未勝利","1勝","2勝","3勝","ｵｰﾌﾟﾝ","OP(L)","Ｇ３","Ｇ２","Ｇ１"],
        "年齢限定": ["なし","２歳","３歳","３上","４上"],
        "馬場状態": ["良","稍","重","不","不明"],
        "競走記号": ["混","牝","国","特","指"],   # 欠損なし想定
        "指定条件": ["指","特"],                 # 後で「なし」を付与
        "限定":     ["混","国"],                  # 後で「なし」を付与
    }
    if pipeline is None:
        FALLBACK["指定条件"] = ["なし"] + FALLBACK["指定条件"]
        FALLBACK["限定"]     = ["なし"] + FALLBACK["限定"]
        return FALLBACK

    try:
        pre = pipeline.named_steps["prep"]
        cat_cols = pre.transformers_[0][2]
        oh = pre.named_transformers_["cat"].named_steps["oh"]
    except Exception:
        FALLBACK["指定条件"] = ["なし"] + FALLBACK["指定条件"]
        FALLBACK["限定"]     = ["なし"] + FALLBACK["限定"]
        return FALLBACK

    result = {}
    for col, cats in zip(cat_cols, oh.categories_):
        vals = [str(v) for v in cats]
        result[col] = vals

    # 指定条件・限定 には常に「なし」を先頭に用意（学習時に欠損があった列）
    for col in ["指定条件", "限定"]:
        vals = list(result.get(col, []))
        if "なし" in vals:
            vals = ["なし"] + [v for v in vals if v != "なし"]
        else:
            vals = ["なし"] + vals
        result[col] = vals

    for k, v in FALLBACK.items():
        result.setdefault(k, v)
    return result

# ---------- 並び順の明示（比較時だけ正規化） ----------
ORDER_PREF = {
    "場所": ["札幌","函館","福島","新潟","東京","中山","中京","京都","阪神","小倉"],
    "天気": ["晴","曇","小雨","雨","小雪","雪","その他"],  # 末尾スペース有りでも strip 比較でマッチ
    "馬場状態": ["良","稍","稍重","重","不","不良","不明"],
    "前走脚質": ["逃げ","先行","中団","ﾏｸﾘ","マクリ","後方"],
    "クラス名": ["新馬","未勝利","1勝","2勝","3勝","ｵｰﾌﾟﾝ","OP(L)","Ｇ３","Ｇ２","Ｇ１"],
    "年齢限定": ["なし","２歳","３歳","３上","４上","2歳","3歳","3歳以上","4歳以上"],
    "指定条件": ["なし","指","特"],
    "限定":     ["なし","混","国"],
}
def _reorder_choices(current: list, desired: list, normalizer=lambda x: x):
    """desired の順を優先。比較は normalizer（例：str.strip）を通す。"""
    rank = {normalizer(v): i for i, v in enumerate(desired)}
    seen = []
    for v in current:
        if v not in seen:
            seen.append(v)
    return sorted(seen, key=lambda x: (rank.get(normalizer(x), 10_000), normalizer(x)))

# ---------- 表示だけ人間語（内部は学習と同じ値） ----------
DISPLAY_MAP = {
    "馬場状態": {"稍":"稍重", "不":"不良"},
    "前走脚質": {"ﾏｸﾘ":"マクリ"},
    "クラス名": {"1勝":"1勝クラス","2勝":"2勝クラス","3勝":"3勝クラス",
                 "ｵｰﾌﾟﾝ":"オープン","OP(L)":"L","Ｇ１":"G1","Ｇ２":"G2","Ｇ３":"G3"},
    "年齢限定": {"２歳":"2歳","３歳":"3歳","３上":"3歳以上","４上":"4歳以上"},
    # 天気は末尾スペースの見た目だけ整える（実値はそのまま）
    "天気": {"晴 ":"晴","曇 ":"曇","雨 ":"雨","雪 ":"雪","小雨 ":"小雨","小雪 ":"小雪"},
    "指定条件": {
        "（若）": "若手騎手",
        "(若)":  "若手騎手",
        "若":    "若手騎手",

        "（特）": "（特指）",
        "(特)":   "（特指）",
        "特":     "（特指）",

        "（指）": "（指定）",
        "(指)":   "（指定）",
        "指":     "（指定）",

        "［指］": "［指定］",
        "[指]":   "［指定］",
    },
    "限定": {
        "混": "（混合）",
        "国": "（国際）",
    },
}
def display_formatter(col):
    dm = DISPLAY_MAP.get(col, {})
    def _fmt(v):
        s = str(v)
        s_clean = s.strip()
        return dm.get(s, dm.get(s_clean, s_clean))  # まず完全一致→なければ strip 版
    return _fmt

# === 競走記号の人間向け表示（A/N, 牝, （指）/［指］/（若）/（特）, 0=なし） ===
def decode_race_symbol(code: str) -> str:
    orig = str(code).strip()
    if orig == "" or orig == "0":
        return "なし"
    s = orig
    parts = []
    # 先頭記号
    if s[0].upper() == "A":
        parts.append("（混合）")
        s = s[1:]
    elif s[0].upper() == "N":
        parts.append("（国際）")
        s = s[1:]
    # 数字部
    digits = "".join(ch for ch in s if ch.isdigit())
    if digits:
        ones = int(digits[-1])                 # 末尾
        tens = int(digits[-2]) if len(digits) >= 2 else 0
        if tens == 2:
            parts.append("牝")
        if   ones == 1: parts.append("（指定）")
        elif ones == 2: parts.append("若手騎手")
        elif ones == 3: parts.append("［指定］")
        elif ones == 4: parts.append("（特指）")
    return "".join(parts) or "なし"

# ===== モデル読み込み（サイドバーなし） =====
pipeline, pipe_src = _load_pipeline(None)
meta, meta_src = _load_meta(None)

thr_default = 0.50
if isinstance(meta, dict):
    thr_default = float(np.clip(meta.get("best_f1_threshold", 0.50), 0.10, 0.90))
threshold = thr_default  # しきい値は画面からは表示・変更しない（固定）

# 学習語彙から候補を構成（並び順も整理：比較は strip）
OPTS = _choices_from_pipeline_exact(pipeline, meta)
for col, order in ORDER_PREF.items():
    if col in OPTS:
        OPTS[col] = _reorder_choices(OPTS[col], order, normalizer=lambda s: str(s).strip())

# ===== 入力フォーム =====
with st.form("inference"):
    st.subheader("判定に使う20項目を入力してください")
    c1, c2, c3 = st.columns(3)

    with c1:
        place       = st.selectbox("場所", OPTS["場所"], format_func=display_formatter("場所"))
        weather     = st.selectbox("天気", OPTS["天気"], format_func=display_formatter("天気"))
        pace_prev   = st.selectbox(
            "前走脚質",
            OPTS["前走脚質"],
            format_func=display_formatter("前走脚質"),
            help=(
                "・逃げ: 2角・3角・4角のどこかで1回でも先頭にいた（マクリを除く）\n"
                "・先行: 出走頭数を3等分して、4角の位置で前から1番目のグループにいた（マクリを除く）\n"
                "・中団: 出走頭数を3等分して、4角の位置で前から2番目のグループにいた\n"
                "・マクリ: 出走頭数を3等分して、2角または3角の位置で前から3番目のグループ、4角の位置で前から1番目のグループにいた\n"
                "・後方: 出走頭数を3等分して、4角の位置で前から3番目のグループにいた"
            ),
        )
        class_name  = st.selectbox("クラス名", OPTS["クラス名"], format_func=display_formatter("クラス名"))
        age_lim     = st.selectbox("年齢限定", OPTS["年齢限定"], format_func=display_formatter("年齢限定"))
        going       = st.selectbox("馬場状態", OPTS["馬場状態"], format_func=display_formatter("馬場状態"))
        race_symbol = st.selectbox("競走記号", OPTS["競走記号"], format_func=decode_race_symbol)

    with c2:
        shitei = st.selectbox("指定条件", OPTS["指定条件"], format_func=display_formatter("指定条件"))

        # 数値は表示しない。内部値は JV コード（10〜29）
        track_jv = st.selectbox(
            "コース形態",
            TRACK_CODE_KEYS,
            index=(TRACK_CODE_KEYS.index(11) if 11 in TRACK_CODE_KEYS else 0),
            format_func=lambda k: TRACK_CODE_LABELS.get(k, ""),
            help="芝/ダート・回り・外/内/直線などを選択"
        )

        blink_code = st.selectbox(
            "ブリンカーの変更",
            BLINKER_OPTIONS,
            format_func=_fmt_blinker,
            help="前走から今走までの装着状況の変化"
        )
        odds        = st.number_input("単勝オッズ", min_value=0.0, value=10.0, step=0.1, format="%.1f")
        career      = st.number_input("キャリア(出走数)", min_value=0, value=5, step=1)
        diff_margin = st.number_input("前走着差タイム（秒, 先着は負）", value=0.0, step=0.1)

        ho9 = st.number_input(
            "前走補9",
            value=81.0,
            step=0.5,
            help=(
                "クラスや距離の差をならして、全体の中でどのくらい強いかを表した"
                "TARGET frontier JVで用いられる指数"
            ),
        )
        st.caption(
            "※2024年の平均値は約81であったため、TARGET frontier JVを利用されていない方は81のままで構いません。"
        )

    with c3:
        gentei   = st.selectbox("限定", OPTS["限定"], format_func=display_formatter("限定"))
        wakuban  = st.number_input("枠番(1-8)", min_value=1, max_value=8, value=1, step=1)
        kinryo   = st.number_input("斤量(kg)", min_value=40.0, max_value=62.0, value=55.0, step=0.5)
        temp_diff = st.number_input("気温差（今走-前走, ℃）", value=0.0, step=0.5)

        agari3f_rank = st.number_input(
            "前走上り3F順",
            min_value=1,
            max_value=30,
            value=10,
            step=1,
            help="前走の上り3Fタイムの順位",
        )

        dist_diff = st.number_input("距離差（今走-前走, m）", min_value=-3000, max_value=3000, value=0, step=100)

    row = {
        "場所": place, "天気": weather, "前走脚質": pace_prev, "クラス名": class_name, "年齢限定": age_lim,
        "馬場状態": going, "競走記号": race_symbol, "指定条件": shitei, "トラックコード(JV)": int(track_jv),
        "ブリンカー変化コード": int(blink_code), "単勝オッズ": float(odds), "キャリア": int(career),
        "前走着差タイム": float(diff_margin), "前走補9": float(ho9), "限定": gentei, "枠番": int(wakuban),
        "斤量": float(kinryo), "気温差": float(temp_diff), "前走上り3F順": int(agari3f_rank), "距離差": int(dist_diff),
    }
    df_in = pd.DataFrame([row], columns=FEATURES)
    submitted = st.form_submit_button("この内容で判定する")

# ===== 予測 =====
if submitted:
    if pipeline is None:
        st.error("学習済み Pipeline が読み込まれていません。フォルダに `pipeline_xgb_3group.pkl` を置いてから実行してください。")
    else:
        try:
            proba = pipeline.predict_proba(df_in)[:, 1]
            p = float(proba[0])
            is_abnormal = (p >= threshold)
            msg = f"推定確率: {p*100:.1f}% → 判定: **{'異常(怪我)' if is_abnormal else '完走(非異常)'}**（しきい値 {threshold:.2f}）"

            # 判定メッセージ（赤 or 緑）
            if is_abnormal:
                st.error(msg)  # 異常は赤
            else:
                st.success(msg)  # 完走は緑

            # ★特徴量重要度に関する説明（カード風に整形）
            st.markdown(
                """
<div style="
    background-color: #f8f9fa;
    color: #000000;
    padding: 0.8rem 1rem;
    border-left: 4px solid #6c757d;
    border-radius: 4px;
    font-size: 0.95rem;
    margin-top: 0.75rem;
">
<span style="font-weight: 700;">このモデルで重要度が高かった主な項目</span><br>
特徴量重要度の分析では、入力項目のうち、次の順で怪我の有無との関連が相対的に強い（重要度が高い）傾向がみられます。<br>
<span style="font-weight: 700;">・場所</span><br>
<span style="font-weight: 700;">・馬場状態</span><br>
<span style="font-weight: 700;">・年齢限定</span><br>
<span style="font-weight: 700;">・クラス名</span><br>
<span style="font-weight: 700;">・前走脚質</span>
</div>
                """,
                unsafe_allow_html=True,
            )

            # ★怪我リスクを下げるためのヒント（これは赤いカード・異常時だけ）
            if is_abnormal:
                st.markdown(
                    """
<div style="
    background-color: #ffffff;
    color: #000000;
    padding: 0.8rem 1rem;
    border-left: 4px solid #dc3545;
    border-radius: 4px;
    font-size: 1.05rem;
    font-weight: 500;
    margin-top: 0.5rem;
">
<span style="font-weight: 700;">怪我リスクを下げるためのヒント</span><br>
過去データでは、ハイペースで上がりが遅い競馬よりも、スローペースで上がりが速い競馬の方が、怪我率が最大で約50％低いグループもあります。<br>
前半のペースを抑え、終い重視の戦法（スローペースでの逃げ・後方待機など）を検討することで、怪我リスクを相対的に抑えやすくなります。
</div>
                    """,
                    unsafe_allow_html=True,
                )

            with st.expander("送信データ（確認）", expanded=False):
                st.dataframe(df_in)

        except Exception as e:
            st.exception(e)
            st.error("学習時の列名・前処理と一致していない可能性があります。学習スクリプトの出力 Pipeline を使ってください。")
