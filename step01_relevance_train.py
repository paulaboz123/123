import argparse, json
import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import precision_recall_curve, average_precision_score

def recall_at_precision(y_true, scores, precision_target):
    p, r, thr = precision_recall_curve(y_true, scores)
    best = 0.0
    best_thr = None
    for i in range(1, len(p)):
        if p[i] >= precision_target and r[i] > best:
            best = float(r[i])
            best_thr = float(thr[i-1])
    return best, best_thr

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train_csv", required=True)
    ap.add_argument("--text_col", default="text")
    ap.add_argument("--rel_col", default="relevant")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--test_size", type=float, default=0.30)
    ap.add_argument("--precision_target", type=float, default=0.90)
    ap.add_argument("--out_model", required=True)
    ap.add_argument("--out_metrics", required=True)
    args = ap.parse_args()

    df = pd.read_csv(args.train_csv)
    df[args.rel_col] = df[args.rel_col].fillna(0).astype(int)

    X = df[args.text_col].astype(str)
    y = df[args.rel_col].astype(int)

    Xtr, Xva, ytr, yva = train_test_split(
        X, y, test_size=args.test_size, random_state=args.seed,
        stratify=y if y.nunique() > 1 else None
    )

    artifacts = {}
    metrics = []

    lr = LogisticRegression(max_iter=5000, class_weight="balanced")
    lr_pipe = Pipeline([
        ("tfidf", TfidfVectorizer(ngram_range=(1,3), min_df=2, max_df=0.95)),
        ("clf", lr),
    ])
    lr_pipe.fit(Xtr, ytr)
    s = lr_pipe.predict_proba(Xva)[:,1]
    apv = float(average_precision_score(yva, s))
    rec, thr = recall_at_precision(yva, s, args.precision_target)
    metrics.append({"model":"tfidf_logreg","pr_auc":apv,f"recall@p>={args.precision_target}":rec,"threshold":thr})
    artifacts["tfidf_logreg"] = lr_pipe

    svc = LinearSVC(class_weight="balanced", max_iter=8000)
    svc_pipe = Pipeline([
        ("tfidf", TfidfVectorizer(ngram_range=(1,3), min_df=2, max_df=0.95)),
        ("clf", svc),
    ])
    svc_pipe.fit(Xtr, ytr)
    cal = CalibratedClassifierCV(svc_pipe, method="sigmoid", cv=3)
    cal.fit(Xtr, ytr)
    s = cal.predict_proba(Xva)[:,1]
    apv = float(average_precision_score(yva, s))
    rec, thr = recall_at_precision(yva, s, args.precision_target)
    metrics.append({"model":"tfidf_linearsvc_cal","pr_auc":apv,f"recall@p>={args.precision_target}":rec,"threshold":thr})
    artifacts["tfidf_linearsvc_cal"] = cal

    try:
        import lightgbm as lgb
        pos = int((ytr==1).sum()); neg = int((ytr==0).sum())
        lgbm = lgb.LGBMClassifier(
            n_estimators=600, learning_rate=0.05, num_leaves=64,
            subsample=0.9, colsample_bytree=0.9, reg_lambda=1.0,
            random_state=args.seed, n_jobs=-1,
            scale_pos_weight=(neg/pos) if pos>0 else 1.0
        )
        lgbm_pipe = Pipeline([
            ("tfidf", TfidfVectorizer(ngram_range=(1,3), min_df=2, max_df=0.95)),
            ("clf", lgbm),
        ])
        lgbm_pipe.fit(Xtr, ytr)
        s = lgbm_pipe.predict_proba(Xva)[:,1]
        apv = float(average_precision_score(yva, s))
        rec, thr = recall_at_precision(yva, s, args.precision_target)
        metrics.append({"model":"tfidf_lgbm","pr_auc":apv,f"recall@p>={args.precision_target}":rec,"threshold":thr})
        artifacts["tfidf_lgbm"] = lgbm_pipe
    except Exception as e:
        print("LightGBM skipped:", str(e))

    def key(m):
        return (m["pr_auc"], m.get(f"recall@p>={args.precision_target}", 0.0))
    best = sorted(metrics, key=key, reverse=True)[0]
    best_name = best["model"]

    payload = {"model": artifacts[best_name], "threshold": best["threshold"], "rel_col": args.rel_col, "text_col": args.text_col}
    joblib.dump(payload, args.out_model)

    with open(args.out_metrics, "w", encoding="utf-8") as f:
        json.dump({"all": metrics, "best": best}, f, indent=2)

    print("DONE step01 best:", best_name, "thr:", best["threshold"])

if __name__ == "__main__":
    main()
