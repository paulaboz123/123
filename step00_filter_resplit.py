import argparse
import pandas as pd
from sklearn.model_selection import train_test_split

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train_csv", required=True)
    ap.add_argument("--test_csv", required=True)
    ap.add_argument("--labels_csv", required=True)
    ap.add_argument("--text_col", default="text")
    ap.add_argument("--demand_col", default="demand_id")
    ap.add_argument("--exclude_col", default="exclude")
    ap.add_argument("--test_size", type=float, default=0.30)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out_train", required=True)
    ap.add_argument("--out_test", required=True)
    args = ap.parse_args()

    train = pd.read_csv(args.train_csv)
    test  = pd.read_csv(args.test_csv)
    labels = pd.read_csv(args.labels_csv)

    train[args.demand_col] = train[args.demand_col].astype(str)
    test[args.demand_col]  = test[args.demand_col].astype(str)
    labels[args.demand_col] = labels[args.demand_col].astype(str)

    df = pd.concat([train, test], ignore_index=True)

    excluded = set(labels.loc[labels[args.exclude_col] == 1, args.demand_col].astype(str))
    df = df[~df[args.demand_col].isin(excluded)].copy()

    counts = df[args.demand_col].value_counts()
    single = set(counts[counts == 1].index)

    df_single = df[df[args.demand_col].isin(single)]
    df_multi  = df[~df[args.demand_col].isin(single)]

    if df_multi[args.demand_col].nunique() <= 1:
        train_clean = df.copy()
        test_clean = df.iloc[0:0].copy()
    else:
        train_multi, test_multi = train_test_split(
            df_multi,
            test_size=args.test_size,
            random_state=args.seed,
            stratify=df_multi[args.demand_col],
        )
        train_clean = pd.concat([train_multi, df_single], ignore_index=True)
        test_clean  = test_multi.copy()

    train_clean.to_csv(args.out_train, index=False)
    test_clean.to_csv(args.out_test, index=False)

    print("DONE step00")
    print("train_rows", len(train_clean), "test_rows", len(test_clean))
    print("train_labels", train_clean[args.demand_col].nunique(), "test_labels", test_clean[args.demand_col].nunique())

if __name__ == "__main__":
    main()
