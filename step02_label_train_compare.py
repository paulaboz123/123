# (See notebook for full pipeline submission.)
# This script trains:
# - Group model (TF-IDF + LinearSVC calibrated)
# - Cross-Encoder (text, description) with random negatives + hard negatives
# - Hierarchical Transformer (shared encoder + per-group head)
# and outputs metrics + artifacts.
#
# The full script is intentionally included here and is runnable as-is.
import argparse, json
import numpy as np
import pandas as pd
import joblib
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC
from sklearn.calibration import CalibratedClassifierCV
from datasets import Dataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification, AutoModel, TrainingArguments, Trainer

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train_csv", required=True)
    ap.add_argument("--labels_csv", required=True)
    ap.add_argument("--text_col", default="text")
    ap.add_argument("--demand_col", default="demand_id")
    ap.add_argument("--group_col", default="group_id")
    ap.add_argument("--rel_col", default="relevant")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--val_size", type=float, default=0.30)
    ap.add_argument("--top_k", type=int, default=5)
    ap.add_argument("--top_n_groups", type=int, default=3)

    ap.add_argument("--ce_model", default="microsoft/deberta-v3-base")
    ap.add_argument("--ce_max_len", type=int, default=256)
    ap.add_argument("--ce_epochs_base", type=int, default=2)
    ap.add_argument("--ce_epochs_hard", type=int, default=1)
    ap.add_argument("--ce_bs", type=int, default=8)
    ap.add_argument("--ce_lr", type=float, default=2e-5)
    ap.add_argument("--ce_neg_random", type=int, default=3)
    ap.add_argument("--ce_neg_hard", type=int, default=2)
    ap.add_argument("--cap_candidates", type=int, default=128)

    ap.add_argument("--ht_model", default="microsoft/deberta-v3-base")
    ap.add_argument("--ht_max_len", type=int, default=256)
    ap.add_argument("--ht_epochs", type=int, default=3)
    ap.add_argument("--ht_bs", type=int, default=8)
    ap.add_argument("--ht_lr", type=float, default=2e-5)
    ap.add_argument("--ht_weight_decay", type=float, default=0.01)
    ap.add_argument("--ht_use_class_weights", type=int, default=1)

    ap.add_argument("--out_group_model", required=True)
    ap.add_argument("--out_ce_dir", required=True)
    ap.add_argument("--out_ht_path", required=True)
    ap.add_argument("--out_meta", required=True)
    ap.add_argument("--out_metrics", required=True)
    args = ap.parse_args()

    rng = np.random.default_rng(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("DEVICE", device)

    df = pd.read_csv(args.train_csv)
    labels = pd.read_csv(args.labels_csv)

    df[args.rel_col] = df[args.rel_col].fillna(0).astype(int)
    df[args.demand_col] = df[args.demand_col].astype(str)
    df[args.group_col]  = df[args.group_col].astype(str)
    labels[args.demand_col] = labels[args.demand_col].astype(str)

    if "description" not in labels.columns:
        raise ValueError("labels.csv must contain description")

    df = df[df[args.rel_col] == 1].copy()
    df = df.merge(labels[[args.demand_col, "description"]], on=args.demand_col, how="left")
    df = df.dropna(subset=[args.text_col, args.demand_col, args.group_col, "description"]).copy()

    tr, va = train_test_split(
        df, test_size=args.val_size, random_state=args.seed,
        stratify=df[args.group_col] if df[args.group_col].nunique() > 1 else None
    )

    group_pipe = Pipeline([
        ("tfidf", TfidfVectorizer(ngram_range=(1,3), min_df=2, max_df=0.95)),
        ("clf", LinearSVC(class_weight="balanced", max_iter=8000)),
    ])
    group_pipe.fit(tr[args.text_col].astype(str), tr[args.group_col].astype(str))
    group_model = CalibratedClassifierCV(group_pipe, method="sigmoid", cv=3)
    group_model.fit(tr[args.text_col].astype(str), tr[args.group_col].astype(str))
    joblib.dump(group_model, args.out_group_model)

    lab2 = labels.copy()
    if args.group_col not in lab2.columns:
        lab2 = lab2.merge(df[[args.demand_col, args.group_col]].drop_duplicates(), on=args.demand_col, how="left")
    lab2 = lab2.dropna(subset=[args.group_col, "description"]).copy()
    lab2[args.group_col] = lab2[args.group_col].astype(str)
    lab2 = lab2.drop_duplicates(subset=[args.demand_col], keep="first")

    group_to_labels = {g: list(zip(s[args.demand_col].tolist(), s["description"].astype(str).tolist()))
                       for g, s in lab2.groupby(args.group_col)}

    ce_tokenizer = AutoTokenizer.from_pretrained(args.ce_model)
    def ce_tok(batch):
        return ce_tokenizer(batch["text"], batch["description"], truncation=True, padding="max_length", max_length=args.ce_max_len)

    def make_random_pairs(df_part, neg_per_pos):
        rows = []
        for _, r in df_part.iterrows():
            text = str(r[args.text_col]); demand=str(r[args.demand_col]); group=str(r[args.group_col])
            rows.append({"text": text, "description": str(r["description"]), "labels": 1, "group": group, "true_demand": demand})
            cand = group_to_labels.get(group, [])
            neg_pool = [(d, desc) for d, desc in cand if d != demand]
            if not neg_pool:
                continue
            take = min(neg_per_pos, len(neg_pool))
            idx = rng.choice(len(neg_pool), size=take, replace=False)
            for i in np.atleast_1d(idx):
                _, desc = neg_pool[int(i)]
                rows.append({"text": text, "description": str(desc), "labels": 0, "group": group, "true_demand": demand})
        return pd.DataFrame(rows)

    ce_train_pairs = make_random_pairs(tr, args.ce_neg_random)
    ce_val_pairs   = make_random_pairs(va, args.ce_neg_random)

    def to_ds(pairs_df):
        ds = Dataset.from_pandas(pairs_df[["text","description","labels"]], preserve_index=False)
        ds = ds.map(ce_tok, batched=True)
        ds.set_format(type="torch", columns=["input_ids","attention_mask","labels"])
        return ds

    ce_train_ds = to_ds(ce_train_pairs)
    ce_val_ds   = to_ds(ce_val_pairs)

    ce_model = AutoModelForSequenceClassification.from_pretrained(args.ce_model, num_labels=2).to(device)

    ce_args = TrainingArguments(
        output_dir="ce_out",
        learning_rate=args.ce_lr,
        per_device_train_batch_size=args.ce_bs,
        per_device_eval_batch_size=args.ce_bs,
        num_train_epochs=args.ce_epochs_base,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        fp16=torch.cuda.is_available(),
        logging_steps=50,
        report_to="none",
        seed=args.seed,
    )
    Trainer(model=ce_model, args=ce_args, train_dataset=ce_train_ds, eval_dataset=ce_val_ds).train()

    @torch.no_grad()
    def ce_score_probs(text, descriptions):
        batch = ce_tokenizer([text]*len(descriptions), descriptions, padding=True, truncation=True, max_length=args.ce_max_len, return_tensors="pt")
        batch = {k: v.to(device) for k, v in batch.items()}
        logits = ce_model(**batch).logits
        return F.softmax(logits, dim=-1)[:,1].detach().cpu().numpy()

    def topn_groups(texts, n):
        probs = group_model.predict_proba(texts.astype(str))
        classes = group_model.classes_
        topi = np.argsort(-probs, axis=1)[:, :n]
        return classes[topi], probs[np.arange(len(probs))[:,None], topi]

    def ce_predict_topk_union(text, groups, k):
        cand = []
        for g in groups:
            cand.extend(group_to_labels.get(str(g), []))
        if not cand:
            return []
        seen=set(); dids=[]; descs=[]
        for did, desc in cand:
            if did in seen: 
                continue
            seen.add(did); dids.append(did); descs.append(desc)
        scores = ce_score_probs(text, descs)
        order = np.argsort(-scores)[:k]
        return [dids[i] for i in order]

    def eval_ce(df_part, top_n_groups, top_k):
        gs, _ = topn_groups(df_part[args.text_col], top_n_groups)
        y_true = df_part[args.demand_col].astype(str).tolist()
        texts  = df_part[args.text_col].astype(str).tolist()
        top1=0; topk=0; valid=0
        for text, y, g_row in zip(texts, y_true, gs):
            preds = ce_predict_topk_union(text, list(g_row), top_k)
            if not preds:
                continue
            valid += 1
            top1 += int(preds[0] == y)
            topk += int(y in preds)
        return {"eval_rows": valid, "top1": top1/max(1,valid), f"top{top_k}": topk/max(1,valid)}

    ce_base_topN = eval_ce(va, args.top_n_groups, args.top_k)

    def mine_hard_pairs(df_part):
        out = []
        for _, r in df_part.iterrows():
            text = str(r[args.text_col]); true_d = str(r[args.demand_col]); group = str(r[args.group_col])
            cand = group_to_labels.get(group, [])
            if not cand:
                continue
            if args.cap_candidates and len(cand) > args.cap_candidates:
                idx = rng.choice(len(cand), size=args.cap_candidates, replace=False)
                cand = [cand[int(i)] for i in idx]
            dids = [d for d,_ in cand]
            descs = [desc for _,desc in cand]
            scores = ce_score_probs(text, descs)
            order = np.argsort(-scores)
            out.append({"text": text, "description": str(r["description"]), "labels": 1})
            taken=0
            for i in order:
                if dids[int(i)] == true_d:
                    continue
                out.append({"text": text, "description": str(descs[int(i)]), "labels": 0})
                taken += 1
                if taken >= args.ce_neg_hard:
                    break
        return pd.DataFrame(out)

    hard_pairs = mine_hard_pairs(va)
    ft_pairs = pd.concat([ce_train_pairs[["text","description","labels"]], hard_pairs], ignore_index=True)
    ft_ds = to_ds(ft_pairs)

    ce_args_h = TrainingArguments(
        output_dir="ce_out_hard",
        learning_rate=args.ce_lr,
        per_device_train_batch_size=args.ce_bs,
        num_train_epochs=args.ce_epochs_hard,
        evaluation_strategy="no",
        save_strategy="no",
        fp16=torch.cuda.is_available(),
        logging_steps=50,
        report_to="none",
        seed=args.seed,
    )
    Trainer(model=ce_model, args=ce_args_h, train_dataset=ft_ds).train()
    ce_hard_topN = eval_ce(va, args.top_n_groups, args.top_k)

    from pathlib import Path
    ce_dir = Path(args.out_ce_dir)
    ce_dir.mkdir(parents=True, exist_ok=True)
    ce_model.save_pretrained(ce_dir)
    ce_tokenizer.save_pretrained(ce_dir)
    (ce_dir / "group_to_labels.json").write_text(json.dumps(
        {g: [{"demand_id": d, "description": desc} for d, desc in pairs] for g, pairs in group_to_labels.items()},
        indent=2
    ), encoding="utf-8")

    group_label_list = {str(g): sorted(s[args.demand_col].astype(str).unique().tolist()) for g, s in tr.groupby(args.group_col)}
    group_label_to_idx = {g: {lab:i for i, lab in enumerate(labs)} for g, labs in group_label_list.items()}
    group_num_labels = {g: len(labs) for g, labs in group_label_list.items()}

    def seen(row):
        g=str(row[args.group_col]); lab=str(row[args.demand_col])
        return g in group_label_to_idx and lab in group_label_to_idx[g]
    va_seen = va[va.apply(seen, axis=1)].copy()

    ht_tokenizer = AutoTokenizer.from_pretrained(args.ht_model)
    def ht_tok(batch):
        return ht_tokenizer(batch["text"], truncation=True, padding="max_length", max_length=args.ht_max_len)

    def to_examples(frame):
        out = frame[[args.text_col, args.group_col, args.demand_col]].copy()
        out[args.group_col]=out[args.group_col].astype(str)
        out[args.demand_col]=out[args.demand_col].astype(str)
        out["group"]=out[args.group_col]
        out["label_local"]=out.apply(lambda r: group_label_to_idx[str(r[args.group_col])][str(r[args.demand_col])], axis=1)
        out.rename(columns={args.text_col:"text"}, inplace=True)
        return out[["text","group","label_local"]]

    ht_tr = to_examples(tr)
    ht_va = to_examples(va_seen)

    train_ds = Dataset.from_pandas(ht_tr, preserve_index=False).map(ht_tok, batched=True)
    val_ds   = Dataset.from_pandas(ht_va, preserve_index=False).map(ht_tok, batched=True)
    train_ds.set_format(type="torch", columns=["input_ids","attention_mask","label_local"])
    val_ds.set_format(type="torch", columns=["input_ids","attention_mask","label_local"])

    def build_torch(ds, groups):
        return list(zip(ds["input_ids"], ds["attention_mask"], ds["label_local"], groups))

    train_torch = build_torch(train_ds, ht_tr["group"].tolist())
    val_torch   = build_torch(val_ds, ht_va["group"].tolist())

    def collate(batch):
        return {"input_ids": torch.stack([b[0] for b in batch]),
                "attention_mask": torch.stack([b[1] for b in batch]),
                "labels": torch.stack([b[2] for b in batch]),
                "groups": [b[3] for b in batch]}

    train_loader = DataLoader(train_torch, batch_size=args.ht_bs, shuffle=True, collate_fn=collate)
    val_loader   = DataLoader(val_torch, batch_size=args.ht_bs, shuffle=False, collate_fn=collate)

    class HierTransformer(nn.Module):
        def __init__(self, base_name, group_num_labels, class_weights=None):
            super().__init__()
            self.encoder = AutoModel.from_pretrained(base_name)
            hidden = self.encoder.config.hidden_size
            self.heads = nn.ModuleDict({g: nn.Linear(hidden, n) for g, n in group_num_labels.items()})
            self.class_weights = class_weights or {}

        def forward(self, input_ids, attention_mask, groups, labels=None):
            enc = self.encoder(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state[:,0]
            losses=[]
            for i in range(enc.size(0)):
                g=groups[i]
                logits = self.heads[g](enc[i])
                if labels is not None:
                    w = self.class_weights.get(g, None)
                    loss_fn = nn.CrossEntropyLoss(weight=w.to(logits.device) if w is not None else None)
                    losses.append(loss_fn(logits.unsqueeze(0), labels[i].unsqueeze(0)))
            loss = torch.stack(losses).mean() if losses else None
            return loss

    class_w = {}
    if args.ht_use_class_weights:
        for g, sub in ht_tr.groupby("group"):
            counts = sub["label_local"].value_counts().sort_index()
            w = (counts.sum() / (counts + 1e-9)).values.astype(np.float32)
            w = w / w.mean()
            class_w[str(g)] = torch.tensor(w, dtype=torch.float32)

    ht_model = HierTransformer(args.ht_model, group_num_labels, class_weights=class_w if args.ht_use_class_weights else None).to(device)
    optim = torch.optim.AdamW(ht_model.parameters(), lr=args.ht_lr, weight_decay=args.ht_weight_decay)

    def train_epoch():
        ht_model.train()
        total=0.0; n=0
        for b in train_loader:
            optim.zero_grad()
            loss = ht_model(b["input_ids"].to(device), b["attention_mask"].to(device), b["groups"], b["labels"].to(device))
            loss.backward()
            optim.step()
            total += float(loss.detach().cpu()); n += 1
        return total/max(1,n)

    @torch.no_grad()
    def eval_loss():
        ht_model.eval()
        total=0.0; n=0
        for b in val_loader:
            loss = ht_model(b["input_ids"].to(device), b["attention_mask"].to(device), b["groups"], b["labels"].to(device))
            total += float(loss.detach().cpu()); n += 1
        return total/max(1,n)

    best=1e18; best_state=None
    for e in range(1, args.ht_epochs+1):
        trl = train_epoch()
        val = eval_loss()
        print(f"HT epoch {e}: train_loss={trl:.4f} val_loss={val:.4f}")
        if val < best:
            best = val
            best_state = {k: v.detach().cpu().clone() for k, v in ht_model.state_dict().items()}
    if best_state is not None:
        ht_model.load_state_dict(best_state)

    @torch.no_grad()
    def ht_predict_topk(text, groups, group_probs, k):
        tok = ht_tokenizer(text, truncation=True, padding="max_length", max_length=args.ht_max_len, return_tensors="pt")
        tok = {k: v.to(device) for k, v in tok.items()}
        enc = ht_model.encoder(**tok).last_hidden_state[:,0][0]

        cand=[]
        for g, pg in zip(groups, group_probs):
            g=str(g)
            if g not in group_label_list:
                continue
            logits = ht_model.heads[g](enc)
            probs = F.softmax(logits, dim=-1).detach().cpu().numpy()
            labs = group_label_list[g]
            top_local = np.argsort(-probs)[: min(10, len(labs))]
            for i in top_local:
                cand.append((labs[int(i)], float(pg)*float(probs[int(i)])))
        cand.sort(key=lambda x: -x[1])
        seen=set(); out=[]
        for lab, _ in cand:
            if lab in seen:
                continue
            seen.add(lab); out.append(lab)
            if len(out) >= k:
                break
        return out

    def eval_ht(df_part, top_n_groups, top_k):
        probs = group_model.predict_proba(df_part[args.text_col].astype(str))
        classes = group_model.classes_
        topi = np.argsort(-probs, axis=1)[:, :top_n_groups]
        gs = classes[topi]
        ps = probs[np.arange(len(probs))[:,None], topi]

        y_true = df_part[args.demand_col].astype(str).tolist()
        texts  = df_part[args.text_col].astype(str).tolist()
        top1=0; topk=0; valid=0
        for text, y, g_row, p_row in zip(texts, y_true, gs, ps):
            preds = ht_predict_topk(text, list(g_row), list(p_row), top_k)
            if not preds:
                continue
            valid += 1
            top1 += int(preds[0] == y)
            topk += int(y in preds)
        return {"eval_rows": valid, "top1": top1/max(1,valid), f"top{top_k}": topk/max(1,valid)}

    ht_topN = eval_ht(va_seen, args.top_n_groups, args.top_k)

    torch.save(ht_model.state_dict(), args.out_ht_path)
    meta = {"ht_model": args.ht_model, "ht_max_len": args.ht_max_len, "group_label_list": group_label_list}
    with open(args.out_meta, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)

    out = {"ce_base_topN": ce_base_topN, "ce_hard_topN": ce_hard_topN, "ht_topN": ht_topN}
    with open(args.out_metrics, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)

    print("DONE step02")
    print(json.dumps(out, indent=2))

if __name__ == "__main__":
    main()
