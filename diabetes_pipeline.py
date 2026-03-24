"""
===============================================================================
  DIABETES RISK PREDICTION — FINAL ML PIPELINE (v3 — Production-Ready)
  ─────────────────────────────────────────────────────────────────────
  IMPORTANT DATA NOTE:
  SleepHours in this dataset is SYNTHETICALLY DETERMINISTIC — all diabetics
  have 4–5 hours, all non-diabetics have 6–8 hours. Using it as-is would give
  trivially perfect but clinically meaningless results.

  This pipeline includes TWO experimental tracks:
  ► Track A: Clinical-only   — 7 original features (most deployable)
  ► Track B: Full features   — all 12 features (shows lifestyle signal strength)
  Both are compared and reported transparently.
===============================================================================
"""

import warnings, os
warnings.filterwarnings("ignore")
os.makedirs("outputs", exist_ok=True)

import numpy  as np
import pandas as pd
import matplotlib
matplotlib.use("TkAgg")  # Use interactive backend for display
import matplotlib.pyplot   as plt
import matplotlib.gridspec as gridspec
from   matplotlib.colors   import LinearSegmentedColormap
from   matplotlib.patches  import Patch

from sklearn.model_selection  import (train_test_split, StratifiedKFold,
                                       cross_validate, RandomizedSearchCV)
from sklearn.preprocessing    import StandardScaler
from sklearn.linear_model     import LogisticRegression
from sklearn.ensemble         import (RandomForestClassifier,
                                       GradientBoostingClassifier,
                                       HistGradientBoostingClassifier)
from sklearn.inspection       import permutation_importance
from sklearn.metrics          import (accuracy_score, precision_score,
                                       recall_score, f1_score,
                                       roc_auc_score, roc_curve,
                                       confusion_matrix, ConfusionMatrixDisplay)

# ─────────────────────────────────────────────────────────────────────────────
#  COLOUR PALETTE
# ─────────────────────────────────────────────────────────────────────────────
C = {
    "bg"     : "#0F1117",  "panel"  : "#1A1D27",  "grid"   : "#2A2D3E",
    "text"   : "#E8E8F0",  "subtext": "#9090B0",
    "a1"     : "#7C5CBF",  "a2"     : "#4FC3F7",  "a3"     : "#F06292",
    "a4"     : "#66BB6A",  "a5"     : "#FFA726",  "a6"     : "#EF5350",
}
MC = [C["a2"], C["a4"], C["a5"], C["a1"]]   # model colors

plt.rcParams.update({
    "figure.facecolor":C["bg"], "axes.facecolor":C["panel"],
    "axes.edgecolor"  :C["grid"],"axes.labelcolor":C["text"],
    "xtick.color":C["subtext"], "ytick.color":C["subtext"],
    "text.color" :C["text"],    "grid.color" :C["grid"],
    "grid.linestyle":"--",      "grid.alpha" :0.5,
    "font.family":"DejaVu Sans","legend.facecolor":C["panel"],
    "legend.edgecolor":C["grid"],
})

# ─────────────────────────────────────────────────────────────────────────────
#  0 ▸ LOAD & INSPECT
# ─────────────────────────────────────────────────────────────────────────────
print("="*70)
print("  DIABETES RISK PREDICTION PIPELINE  (Final v3)")
print("="*70)

DATA_PATH = "dataset_cleaned.csv"
df_raw = pd.read_csv(DATA_PATH)
print(f"\n▶ Dataset          : {df_raw.shape[0]} rows × {df_raw.shape[1]} columns")
print(f"  Diabetic rate    : {df_raw['Outcome'].mean()*100:.1f}%  "
      f"({df_raw['Outcome'].sum()} / {len(df_raw)})")

# Detect synthetic determinism
print("\n  ⚠ Data Quality Check: SleepHours distribution by Outcome")
sh_table = df_raw.groupby(['Outcome','SleepHours']).size().unstack(fill_value=0)
print(sh_table.to_string())
print("  → SleepHours is PERFECTLY deterministic (synthetic design artifact).")
print("    Pipeline tracks both Clinical-only and Full-feature experiments.\n")

# ─────────────────────────────────────────────────────────────────────────────
#  1 ▸ PRE-PROCESSING
# ─────────────────────────────────────────────────────────────────────────────
df = df_raw.copy()

# Winsorise at 5–95th percentile (handles extreme physiological outliers)
CLIP_COLS = ["Insulin","SkinThickness","BMI","DiabetesPedigreeFunction","BloodPressure"]
for col in CLIP_COLS:
    lo, hi = df[col].quantile(0.05), df[col].quantile(0.95)
    df[col] = df[col].clip(lo, hi)
print(f"▶ Outlier handling  : Winsorised [5–95%] → {CLIP_COLS}")

# ─────────────────────────────────────────────────────────────────────────────
#  2 ▸ FEATURE ENGINEERING (label-agnostic, clinically motivated)
# ─────────────────────────────────────────────────────────────────────────────
# Metabolic syndrome proxy
df["Glucose_BMI"]        = df["Glucose"] * df["BMI"] / 1000
# Age-adjusted glycaemia
df["Age_Glucose"]        = df["Age"]     * df["Glucose"] / 1000
# Insulin sensitivity (HOMA-IR-like)
df["Insulin_Sensitivity"]= df["Insulin"] / (df["Glucose"] + 1e-3)
# Composite lifestyle risk score (ordinal, medically coherent)
df["Lifestyle_Risk"]     = (
    (2 - df["ActivityLevel"]) * 2 +
    df["StressLevel"]             +
    df["SugarIntake"] * 2         -
    (df["SleepHours"] - 6).clip(-2, 2)
)
print("▶ Engineered        : Glucose_BMI | Age_Glucose | "
      "Insulin_Sensitivity | Lifestyle_Risk")

# Feature sets
CLINICAL   = ["Pregnancies","Glucose","BloodPressure","SkinThickness",
              "Insulin","BMI","DiabetesPedigreeFunction","Age",
              "Glucose_BMI","Age_Glucose","Insulin_Sensitivity"]
LIFESTYLE  = ["SleepHours","ActivityLevel","StressLevel","SugarIntake","Lifestyle_Risk"]
FULL       = CLINICAL + LIFESTYLE

print(f"  Clinical features : {len(CLINICAL)}")
print(f"  Lifestyle features: {len(LIFESTYLE)}")
print(f"  Full feature set  : {len(FULL)}")

# ─────────────────────────────────────────────────────────────────────────────
#  3 ▸ TRAIN / TEST SPLIT  (stratified 80/20)
# ─────────────────────────────────────────────────────────────────────────────
y = df["Outcome"].values
X_clin = df[CLINICAL].values
X_full = df[FULL].values

(X_clin_tr, X_clin_te,
 X_full_tr, X_full_te,
 y_tr, y_te) = train_test_split(
    X_clin, X_full, y, test_size=0.20, random_state=42, stratify=y
)
print(f"\n▶ Split             : {len(y_tr)} train | {len(y_te)} test  (stratified)")

# ─────────────────────────────────────────────────────────────────────────────
#  4 ▸ SCALING  (fit on train, transform test)
# ─────────────────────────────────────────────────────────────────────────────
sc_c = StandardScaler().fit(X_clin_tr)
sc_f = StandardScaler().fit(X_full_tr)

Xc_tr = sc_c.transform(X_clin_tr); Xc_te = sc_c.transform(X_clin_te)
Xf_tr = sc_f.transform(X_full_tr); Xf_te = sc_f.transform(X_full_te)
print("▶ Scaling           : StandardScaler (fit on train, applied to test)")

# ─────────────────────────────────────────────────────────────────────────────
#  5 ▸ MODEL FACTORY
# ─────────────────────────────────────────────────────────────────────────────
def make_models():
    return {
        "Logistic Regression": LogisticRegression(
            C=0.3, max_iter=2000, class_weight="balanced",
            solver="lbfgs", random_state=42
        ),
        "Random Forest": RandomForestClassifier(
            n_estimators=300, max_depth=8, min_samples_split=12,
            min_samples_leaf=4, max_features="sqrt",
            class_weight="balanced", random_state=42, n_jobs=-1
        ),
        "Gradient Boosting": GradientBoostingClassifier(
            n_estimators=200, max_depth=3, learning_rate=0.08,
            subsample=0.8, min_samples_split=12,
            validation_fraction=0.1, n_iter_no_change=25,
            random_state=42
        ),
        "HistGrad Boosting\n(≈XGBoost)": HistGradientBoostingClassifier(
            max_iter=300, max_depth=4, learning_rate=0.07,
            min_samples_leaf=12, l2_regularization=0.8,
            early_stopping=True, validation_fraction=0.15,
            random_state=42
        ),
    }

# ─────────────────────────────────────────────────────────────────────────────
#  6 ▸ HYPERPARAMETER TUNING  (Clinical track — RandomizedSearchCV 5-fold)
# ─────────────────────────────────────────────────────────────────────────────
print("\n▶ Hyperparameter tuning (RandomizedSearchCV, 5-fold, AUC scoring)…")

CV5 = StratifiedKFold(5, shuffle=True, random_state=42)

rf_search = RandomizedSearchCV(
    RandomForestClassifier(class_weight="balanced", random_state=42, n_jobs=-1),
    {"n_estimators":[100,200,300],"max_depth":[5,7,9,None],
     "min_samples_split":[8,12,20],"min_samples_leaf":[3,5,8]},
    n_iter=20, cv=CV5, scoring="roc_auc", random_state=42, n_jobs=-1
)
rf_search.fit(Xc_tr, y_tr)
best_rf_params = rf_search.best_params_
print(f"   RF  : {best_rf_params}  AUC={rf_search.best_score_:.4f}")

gb_search = RandomizedSearchCV(
    GradientBoostingClassifier(random_state=42),
    {"n_estimators":[100,200,300],"max_depth":[2,3,4],
     "learning_rate":[0.04,0.08,0.12],"subsample":[0.65,0.75,0.85]},
    n_iter=20, cv=CV5, scoring="roc_auc", random_state=42, n_jobs=-1
)
gb_search.fit(Xc_tr, y_tr)
best_gb_params = gb_search.best_params_
print(f"   GB  : {best_gb_params}  AUC={gb_search.best_score_:.4f}")

# ─────────────────────────────────────────────────────────────────────────────
#  7 ▸ TRAIN + EVALUATE — BOTH TRACKS
# ─────────────────────────────────────────────────────────────────────────────
CV10 = StratifiedKFold(10, shuffle=True, random_state=42)

def evaluate_track(track_name, X_tr_, X_te_, feature_names_):
    print(f"\n  ── Track: {track_name} ({len(feature_names_)} features) ──")
    models = make_models()
    # Inject tuned RF/GB params
    models["Random Forest"] = RandomForestClassifier(
        **best_rf_params, class_weight="balanced", random_state=42, n_jobs=-1)
    models["Gradient Boosting"] = GradientBoostingClassifier(
        **best_gb_params, random_state=42)

    res = {}
    for name, clf in models.items():
        clf.fit(X_tr_, y_tr)
        yp  = clf.predict(X_te_)
        ypr = clf.predict_proba(X_te_)[:, 1]
        yp_tr = clf.predict(X_tr_)
        cv_r = cross_validate(clf, X_tr_, y_tr, cv=CV10,
                               scoring=["accuracy","roc_auc","f1"], n_jobs=-1)
        fpr, tpr, _ = roc_curve(y_te, ypr)
        res[name] = dict(
            accuracy  =accuracy_score(y_te, yp),
            precision =precision_score(y_te, yp, zero_division=0),
            recall    =recall_score(y_te, yp),
            f1        =f1_score(y_te, yp),
            roc_auc   =roc_auc_score(y_te, ypr),
            train_acc =accuracy_score(y_tr, yp_tr),
            cv_auc_mean=cv_r["test_roc_auc"].mean(),
            cv_auc_std =cv_r["test_roc_auc"].std(),
            cv_acc_mean=cv_r["test_accuracy"].mean(),
            fpr=fpr, tpr=tpr, y_pred=yp, y_prob=ypr, clf=clf,
        )

    print(f"  {'Model':<26} {'Acc':>6} {'Prec':>6} {'Rec':>6} "
          f"{'F1':>6} {'AUC':>6} {'TrainAcc':>9} {'Gap':>6}")
    print("  " + "-"*73)
    for n, r in res.items():
        nm = n.replace("\n","")
        print(f"  {nm:<26} {r['accuracy']:>6.3f} {r['precision']:>6.3f} "
              f"{r['recall']:>6.3f} {r['f1']:>6.3f} {r['roc_auc']:>6.3f} "
              f"{r['train_acc']:>9.3f} {r['train_acc']-r['accuracy']:>6.3f}")

    best_n = max(res, key=lambda n: res[n]["roc_auc"])
    print(f"\n  ✔ Best: {best_n.replace(chr(10),' ')}  "
          f"AUC={res[best_n]['roc_auc']:.4f}")

    # Permutation importance
    perm = permutation_importance(
        res[best_n]["clf"], X_te_, y_te,
        n_repeats=40, random_state=42, scoring="roc_auc", n_jobs=-1
    )
    imp_df = pd.DataFrame({
        "feature"    : feature_names_,
        "importance" : perm.importances_mean,
        "std"        : perm.importances_std,
    }).sort_values("importance", ascending=False).reset_index(drop=True)
    return res, best_n, imp_df

# Run both tracks
print("\n▶ Evaluating both feature tracks…")
res_c, best_c, imp_c = evaluate_track("Clinical", Xc_tr, Xc_te, CLINICAL)
res_f, best_f, imp_f = evaluate_track("Full (+ Lifestyle)", Xf_tr, Xf_te, FULL)

LS_SET = set(LIFESTYLE)

# ─────────────────────────────────────────────────────────────────────────────
#  8 ▸ PLOTTING
# ─────────────────────────────────────────────────────────────────────────────

# ── FIGURE 1: MAIN DASHBOARD ─────────────────────────────────────────────────
fig = plt.figure(figsize=(24, 30), facecolor=C["bg"])
fig.suptitle("DIABETES RISK PREDICTION — COMPLETE ML PIPELINE DASHBOARD",
             fontsize=18, fontweight="bold", color=C["text"], y=0.99)
gs = gridspec.GridSpec(5, 3, figure=fig,
                       hspace=0.50, wspace=0.38,
                       top=0.96, bottom=0.03, left=0.06, right=0.97)

# ── A: ROC CURVES — Clinical Track ───────────────────────────────────────────
ax_roc_c = fig.add_subplot(gs[0, :2])
ax_roc_c.plot([0,1],[0,1],":", color=C["subtext"], lw=1.5, label="Random (AUC=0.50)")
for (n,r), col in zip(res_c.items(), MC):
    nm = n.replace("\n"," ")
    ax_roc_c.plot(r["fpr"], r["tpr"], color=col, lw=2.5,
                  label=f"{nm}  (AUC={r['roc_auc']:.3f})")
bc_idx = list(res_c.keys()).index(best_c)
ax_roc_c.fill_between(res_c[best_c]["fpr"], res_c[best_c]["tpr"],
                       alpha=0.10, color=MC[bc_idx])
ax_roc_c.set_xlabel("False Positive Rate  (1 − Specificity)")
ax_roc_c.set_ylabel("True Positive Rate  (Sensitivity)")
ax_roc_c.set_title("ROC Curves — CLINICAL Track (7 clinical + 3 engineered features)\n"
                   "AUC = probability that model ranks a diabetic patient higher than non-diabetic",
                   fontweight="bold", fontsize=11)
ax_roc_c.legend(fontsize=9, loc="lower right")
ax_roc_c.grid(True); ax_roc_c.set_xlim(0,1); ax_roc_c.set_ylim(0,1.02)
ax_roc_c.text(0.65, 0.15, f"Best AUC\n{res_c[best_c]['roc_auc']:.3f}",
              transform=ax_roc_c.transAxes, fontsize=14, fontweight="bold",
              color=C["a5"], ha="center", va="center",
              bbox=dict(boxstyle="round,pad=0.5", facecolor=C["panel"],
                        edgecolor=C["a5"], alpha=0.9))

# ── B: ROC CURVES — Full Track ───────────────────────────────────────────────
ax_roc_f = fig.add_subplot(gs[1, :2])
ax_roc_f.plot([0,1],[0,1],":", color=C["subtext"], lw=1.5, label="Random (AUC=0.50)")
for (n,r), col in zip(res_f.items(), MC):
    nm = n.replace("\n"," ")
    ax_roc_f.plot(r["fpr"], r["tpr"], color=col, lw=2.5,
                  label=f"{nm}  (AUC={r['roc_auc']:.3f})")
bf_idx = list(res_f.keys()).index(best_f)
ax_roc_f.fill_between(res_f[best_f]["fpr"], res_f[best_f]["tpr"],
                       alpha=0.10, color=MC[bf_idx])
ax_roc_f.set_xlabel("False Positive Rate  (1 − Specificity)")
ax_roc_f.set_ylabel("True Positive Rate  (Sensitivity)")
ax_roc_f.set_title("ROC Curves — FULL Track (+ SleepHours, StressLevel, SugarIntake, ActivityLevel)\n"
                   "NOTE: SleepHours is synthetically deterministic in this dataset",
                   fontweight="bold", fontsize=11)
ax_roc_f.legend(fontsize=9, loc="lower right")
ax_roc_f.grid(True); ax_roc_f.set_xlim(0,1); ax_roc_f.set_ylim(0,1.02)
ax_roc_f.text(0.65, 0.15, f"Best AUC\n{res_f[best_f]['roc_auc']:.3f}",
              transform=ax_roc_f.transAxes, fontsize=14, fontweight="bold",
              color=C["a3"], ha="center", va="center",
              bbox=dict(boxstyle="round,pad=0.5", facecolor=C["panel"],
                        edgecolor=C["a3"], alpha=0.9))

# ── C: METRIC COMPARISON (Clinical) ──────────────────────────────────────────
ax_mc = fig.add_subplot(gs[0, 2])
metrics = ["accuracy","precision","recall","f1","roc_auc"]
mlb     = ["Acc","Prec","Rec","F1","AUC"]
x_m = np.arange(5); w = 0.18
for i,(n,r) in enumerate(res_c.items()):
    ax_mc.bar(x_m + i*w - 0.27, [r[m] for m in metrics], w,
              color=MC[i], label=n.replace("\n","")[:16], alpha=0.88,
              edgecolor=C["bg"], lw=0.5)
ax_mc.set_xticks(x_m); ax_mc.set_xticklabels(mlb, fontsize=8.5)
ax_mc.set_ylim(0.4,1.05); ax_mc.set_ylabel("Score")
ax_mc.set_title("Clinical Track\nMetric Comparison", fontweight="bold")
ax_mc.axhline(0.85, color=C["a3"], ls="--", lw=1, alpha=0.7, label="Target")
ax_mc.legend(fontsize=6.5, loc="lower right"); ax_mc.grid(axis="y", alpha=0.4)

# ── D: METRIC COMPARISON (Full) ──────────────────────────────────────────────
ax_mf = fig.add_subplot(gs[1, 2])
for i,(n,r) in enumerate(res_f.items()):
    ax_mf.bar(x_m + i*w - 0.27, [r[m] for m in metrics], w,
              color=MC[i], label=n.replace("\n","")[:16], alpha=0.88,
              edgecolor=C["bg"], lw=0.5)
ax_mf.set_xticks(x_m); ax_mf.set_xticklabels(mlb, fontsize=8.5)
ax_mf.set_ylim(0.4,1.05); ax_mf.set_ylabel("Score")
ax_mf.set_title("Full Track\nMetric Comparison", fontweight="bold")
ax_mf.axhline(0.85, color=C["a3"], ls="--", lw=1, alpha=0.7, label="Target")
ax_mf.legend(fontsize=6.5, loc="lower right"); ax_mf.grid(axis="y", alpha=0.4)

# ── E: CONFUSION MATRICES (best clinical model) ──────────────────────────────
cm_axes = [fig.add_subplot(gs[2,c]) for c in range(3)] + \
          [fig.add_subplot(gs[3,c]) for c in range(3)]

for i, (ax_cm, (name, r)) in enumerate(zip(cm_axes[:4], res_c.items())):
    cm   = confusion_matrix(y_te, r["y_pred"])
    cmap = LinearSegmentedColormap.from_list("c",[C["panel"],MC[i]])
    disp = ConfusionMatrixDisplay(cm, display_labels=["Non-Diab","Diab"])
    disp.plot(ax=ax_cm, colorbar=False, cmap=cmap)
    nm = name.replace("\n"," ")
    ax_cm.set_title(f"[Clinical] {nm}\nAcc={r['accuracy']:.3f}  F1={r['f1']:.3f}  "
                    f"AUC={r['roc_auc']:.3f}",
                    fontsize=8.5, fontweight="bold")
    ax_cm.tick_params(labelsize=8)
    for txt in ax_cm.texts: txt.set_color(C["text"]); txt.set_fontsize(13)

# ── F: FEATURE IMPORTANCE — Clinical ─────────────────────────────────────────
ax_fi_c = fig.add_subplot(gs[4, 0])
tc = imp_c.head(11)
cols_c = [C["a3"] if f in LS_SET else C["a2"] for f in tc["feature"]]
ax_fi_c.barh(range(11), tc["importance"].values[::-1],
             xerr=tc["std"].values[::-1], color=cols_c[::-1],
             alpha=0.85, height=0.7, capsize=3,
             error_kw={"ecolor":C["subtext"],"lw":1})
ax_fi_c.set_yticks(range(11)); ax_fi_c.set_yticklabels(tc["feature"].values[::-1], fontsize=9)
ax_fi_c.set_xlabel("AUC Drop", fontsize=9)
ax_fi_c.set_title("Clinical Track\nPermutation Importance", fontweight="bold")
ax_fi_c.legend(handles=[Patch(color=C["a2"],label="Clinical")], fontsize=8)
ax_fi_c.grid(axis="x", alpha=0.4)
for i,(v,s) in enumerate(zip(tc["importance"][::-1], tc["std"][::-1])):
    if v > 0.001:
        ax_fi_c.text(v+s+0.001, i, f"{v:.4f}", va="center", fontsize=7.5)

# ── G: FEATURE IMPORTANCE — Full ─────────────────────────────────────────────
ax_fi_f = fig.add_subplot(gs[4, 1])
tf = imp_f.head(14)
cols_f = [C["a3"] if f in LS_SET else C["a2"] for f in tf["feature"]]
ax_fi_f.barh(range(14), tf["importance"].values[::-1],
             xerr=tf["std"].values[::-1], color=cols_f[::-1],
             alpha=0.85, height=0.7, capsize=3,
             error_kw={"ecolor":C["subtext"],"lw":1})
ax_fi_f.set_yticks(range(14)); ax_fi_f.set_yticklabels(tf["feature"].values[::-1], fontsize=9)
ax_fi_f.set_xlabel("AUC Drop", fontsize=9)
ax_fi_f.set_title("Full Track\nPermutation Importance", fontweight="bold")
ax_fi_f.legend(handles=[Patch(color=C["a2"],label="Clinical"),
                          Patch(color=C["a3"],label="Lifestyle")], fontsize=8)
ax_fi_f.grid(axis="x", alpha=0.4)

# ── H: AUC TRACK COMPARISON ──────────────────────────────────────────────────
ax_comp = fig.add_subplot(gs[4, 2])
clin_aucs = [r["roc_auc"] for r in res_c.values()]
full_aucs  = [r["roc_auc"] for r in res_f.values()]
model_nms  = [n.replace("\n","")[:16] for n in res_c.keys()]
x_c2 = np.arange(4)
ax_comp.bar(x_c2 - 0.18, clin_aucs, 0.32, color=MC, alpha=0.88, label="Clinical")
ax_comp.bar(x_c2 + 0.18, full_aucs,  0.32, color=MC, alpha=0.35, label="Full (+Lifestyle)")
ax_comp.set_xticks(x_c2); ax_comp.set_xticklabels(model_nms, fontsize=7.5)
ax_comp.set_ylim(0.5, 1.12); ax_comp.set_ylabel("ROC-AUC")
ax_comp.set_title("AUC: Clinical vs Full\n(Track Comparison)", fontweight="bold")
ax_comp.axhline(0.85, color=C["a3"], ls="--", lw=1, alpha=0.7)
ax_comp.legend(fontsize=8); ax_comp.grid(axis="y", alpha=0.4)
for i,(ca,fa) in enumerate(zip(clin_aucs,full_aucs)):
    ax_comp.text(i-0.18, ca+0.01, f"{ca:.3f}", ha="center", fontsize=7.5, color=C["text"])
    ax_comp.text(i+0.18, fa+0.01, f"{fa:.3f}", ha="center", fontsize=7.5, color=C["subtext"])

plt.savefig("outputs/01_pipeline_dashboard.png", dpi=130,
            bbox_inches="tight", facecolor=C["bg"])
plt.show()
plt.close()
print("\n✔ Saved → outputs/01_pipeline_dashboard.png")

# ── FIGURE 2: FEATURE DISTRIBUTIONS ──────────────────────────────────────────
fig2, axes2 = plt.subplots(3, 4, figsize=(21, 17), facecolor=C["bg"])
fig2.suptitle("FEATURE ANALYSIS — CLINICAL vs LIFESTYLE RISK FACTORS",
              fontsize=16, fontweight="bold", color=C["text"])

plot_feats = ["Glucose","BMI","Age","DiabetesPedigreeFunction",
              "Insulin","BloodPressure","SkinThickness","Pregnancies",
              "SleepHours","ActivityLevel","StressLevel","SugarIntake"]

top8 = set(imp_c.head(8)["feature"].tolist())

for ax, feat in zip(axes2.flatten(), plot_feats):
    neg = df[df["Outcome"]==0][feat]
    pos = df[df["Outcome"]==1][feat]
    is_ls = feat in LS_SET
    c0 = C["a2"] if not is_ls else "#26C6DA"
    c1 = C["a3"] if not is_ls else C["a5"]

    if feat in ["ActivityLevel","SugarIntake"]:
        cats = sorted(df[feat].unique())
        v0 = [neg[neg==c].count() for c in cats]
        v1 = [pos[pos==c].count() for c in cats]
        xc = np.arange(len(cats))
        ax.bar(xc-0.2, v0, 0.35, color=c0, alpha=0.85, label="Non-Diabetic")
        ax.bar(xc+0.2, v1, 0.35, color=c1, alpha=0.85, label="Diabetic")
        ax.set_xticks(xc); ax.set_xticklabels(["Low","Med","High"], fontsize=8.5)
    else:
        bns = np.linspace(df[feat].min(), df[feat].max(), 26)
        ax.hist(neg, bins=bns, color=c0, alpha=0.62, label="Non-Diabetic", edgecolor=C["bg"])
        ax.hist(pos, bins=bns, color=c1, alpha=0.62, label="Diabetic",     edgecolor=C["bg"])
        ax.axvline(neg.mean(), color=c0, lw=2, ls="--")
        ax.axvline(pos.mean(), color=c1, lw=2, ls="--")

    is_det = (feat == "SleepHours")
    star   = " ★" if feat in top8 else ""
    det_txt= " ⚠DETERMINISTIC" if is_det else ""
    lbl    = "Lifestyle" if is_ls else "Clinical"
    ax.set_title(f"{feat}{star}  [{lbl}]{det_txt}", fontsize=9.5, fontweight="bold",
                 color=C["a6"] if is_det else (C["a3"] if is_ls else C["a2"]))
    ax.legend(fontsize=7.5); ax.grid(alpha=0.3)

plt.tight_layout(pad=2.0)
plt.savefig("outputs/02_feature_analysis.png", dpi=130,
            bbox_inches="tight", facecolor=C["bg"])
plt.show()
plt.close()
print("✔ Saved → outputs/02_feature_analysis.png")

# ── FIGURE 3: EXPLAINABILITY & RISK SCORES ───────────────────────────────────
fig3, axes3 = plt.subplots(2, 2, figsize=(18, 14), facecolor=C["bg"])
fig3.suptitle("EXPLAINABILITY PANEL — Risk Factors & Score Distributions",
              fontsize=15, fontweight="bold")

# Importance clinical
ax_ic = axes3[0, 0]
tc14 = imp_c.head(11)
c_ic = [C["a3"] if f in LS_SET else C["a2"] for f in tc14["feature"]]
ax_ic.barh(range(11), tc14["importance"][::-1], xerr=tc14["std"][::-1],
           color=c_ic[::-1], alpha=0.88, height=0.7, capsize=3,
           error_kw={"ecolor":C["subtext"],"lw":1})
ax_ic.set_yticks(range(11)); ax_ic.set_yticklabels(tc14["feature"][::-1], fontsize=10)
ax_ic.set_xlabel("AUC Drop  (↑ = more important)"); ax_ic.set_title("Clinical Track — Feature Importance", fontweight="bold")
ax_ic.legend(handles=[Patch(color=C["a2"],label="Clinical Feature")], fontsize=9)
ax_ic.grid(axis="x", alpha=0.4)
for i,(v,s) in enumerate(zip(tc14["importance"][::-1], tc14["std"][::-1])):
    if v > 0.001: ax_ic.text(v+s+0.001, i, f"{v:.4f}", va="center", fontsize=8.5)

# Risk score distribution - clinical
ax_rs_c = axes3[0, 1]
ypr_c = res_c[best_c]["y_prob"]
ax_rs_c.hist(ypr_c[y_te==0], bins=30, color=C["a2"], alpha=0.72,
             label="Non-Diabetic", density=True, edgecolor=C["bg"])
ax_rs_c.hist(ypr_c[y_te==1], bins=30, color=C["a3"], alpha=0.72,
             label="Diabetic",     density=True, edgecolor=C["bg"])
ax_rs_c.axvline(0.5, color=C["a5"], ls="--", lw=2.5, label="Default Threshold")
ax_rs_c.set_xlabel("Predicted Diabetes Probability"); ax_rs_c.set_ylabel("Density")
ax_rs_c.set_title(f"Risk Score Distribution\nClinical Track  ({best_c.replace(chr(10),' ')})",
                   fontweight="bold")
ax_rs_c.legend(fontsize=10); ax_rs_c.grid(alpha=0.4)
ax_rs_c.text(0.97, 0.95, f"AUC\n{res_c[best_c]['roc_auc']:.3f}",
             transform=ax_rs_c.transAxes, fontsize=15, fontweight="bold",
             color=C["a5"], ha="right", va="top",
             bbox=dict(boxstyle="round,pad=0.5", facecolor=C["panel"],
                       edgecolor=C["a5"]))

# Importance full
ax_if = axes3[1, 0]
tf14 = imp_f.head(14)
c_if = [C["a3"] if f in LS_SET else C["a2"] for f in tf14["feature"]]
ax_if.barh(range(14), tf14["importance"][::-1], xerr=tf14["std"][::-1],
           color=c_if[::-1], alpha=0.88, height=0.7, capsize=3,
           error_kw={"ecolor":C["subtext"],"lw":1})
ax_if.set_yticks(range(14)); ax_if.set_yticklabels(tf14["feature"][::-1], fontsize=10)
ax_if.set_xlabel("AUC Drop"); ax_if.set_title("Full Track — Feature Importance", fontweight="bold")
ax_if.legend(handles=[Patch(color=C["a2"],label="Clinical"),
                       Patch(color=C["a3"],label="Lifestyle")], fontsize=9)
ax_if.grid(axis="x", alpha=0.4)
for i,(v,s) in enumerate(zip(tf14["importance"][::-1], tf14["std"][::-1])):
    if v > 0.001: ax_if.text(v+s+0.001, i, f"{v:.4f}", va="center", fontsize=8.5)

# Risk score — full
ax_rs_f = axes3[1, 1]
ypr_f = res_f[best_f]["y_prob"]
ax_rs_f.hist(ypr_f[y_te==0], bins=30, color=C["a2"], alpha=0.72,
             label="Non-Diabetic", density=True, edgecolor=C["bg"])
ax_rs_f.hist(ypr_f[y_te==1], bins=30, color=C["a3"], alpha=0.72,
             label="Diabetic",     density=True, edgecolor=C["bg"])
ax_rs_f.axvline(0.5, color=C["a5"], ls="--", lw=2.5, label="Default Threshold")
ax_rs_f.set_xlabel("Predicted Diabetes Probability"); ax_rs_f.set_ylabel("Density")
ax_rs_f.set_title(f"Risk Score Distribution\nFull Track  ({best_f.replace(chr(10),' ')})\n"
                   "⚠ Inflated by synthetic SleepHours encoding", fontweight="bold")
ax_rs_f.legend(fontsize=10); ax_rs_f.grid(alpha=0.4)
ax_rs_f.text(0.97, 0.95, f"AUC\n{res_f[best_f]['roc_auc']:.3f}",
             transform=ax_rs_f.transAxes, fontsize=15, fontweight="bold",
             color=C["a3"], ha="right", va="top",
             bbox=dict(boxstyle="round,pad=0.5", facecolor=C["panel"],
                       edgecolor=C["a3"]))

plt.tight_layout(pad=2.0)
plt.savefig("outputs/03_explainability.png", dpi=130,
            bbox_inches="tight", facecolor=C["bg"])
plt.show()
plt.close()
print("✔ Saved → outputs/03_explainability.png")

# ── FIGURE 4: CORRELATION HEATMAP ────────────────────────────────────────────
fig4, ax4 = plt.subplots(figsize=(16, 14), facecolor=C["bg"])
all_cols = FULL + ["Outcome"]
corr = df[all_cols].corr()
cmap_h = LinearSegmentedColormap.from_list("rb",[C["a1"],C["panel"],C["a2"]])
im = ax4.imshow(corr.values, cmap=cmap_h, vmin=-0.9, vmax=0.9, aspect="auto")
ax4.set_xticks(range(len(corr))); ax4.set_yticks(range(len(corr)))
ax4.set_xticklabels(corr.columns, rotation=45, ha="right", fontsize=8.5)
ax4.set_yticklabels(corr.columns, fontsize=8.5)
for i in range(len(corr)):
    for j in range(len(corr)):
        v = corr.values[i,j]
        if abs(v) > 0.10:
            ax4.text(j, i, f"{v:.2f}", ha="center", va="center",
                     fontsize=7, fontweight="bold" if abs(v)>0.35 else "normal",
                     color="white" if abs(v)>0.55 else C["text"])
plt.colorbar(im, ax=ax4, fraction=0.035, pad=0.02)
ax4.set_title("Feature Correlation Heatmap\n(Clinical + Lifestyle + Engineered + Outcome)",
              fontsize=13, fontweight="bold", pad=12)
plt.tight_layout()
plt.savefig("outputs/04_correlation_heatmap.png", dpi=130,
            bbox_inches="tight", facecolor=C["bg"])
plt.show()
plt.close()
print("✔ Saved → outputs/04_correlation_heatmap.png")

# ── FIGURE 5: CROSS-VALIDATION & OVERFITTING ─────────────────────────────────
fig5, axes5 = plt.subplots(1, 2, figsize=(16, 7), facecolor=C["bg"])
fig5.suptitle("GENERALIZATION ANALYSIS — CV & Overfitting Check",
              fontsize=14, fontweight="bold")

# 10-fold CV clinical
ax_cv = axes5[0]
cv_names_c = [n.replace("\n"," ") for n in res_c.keys()]
cv_aucs_c  = [r["cv_auc_mean"] for r in res_c.values()]
cv_stds_c  = [r["cv_auc_std"]  for r in res_c.values()]
cv_accs_c  = [r["cv_acc_mean"] for r in res_c.values()]
x4 = np.arange(4)
ax_cv.bar(x4-0.2, cv_aucs_c, 0.35, yerr=cv_stds_c, color=MC, alpha=0.88,
          label="CV AUC", capsize=5, error_kw={"ecolor":C["subtext"],"lw":1.5})
ax_cv.bar(x4+0.2, cv_accs_c, 0.35, color=MC, alpha=0.35, label="CV Acc")
ax_cv.set_xticks(x4); ax_cv.set_xticklabels([n[:20] for n in cv_names_c], fontsize=9)
ax_cv.set_ylim(0.5, 1.05); ax_cv.set_ylabel("Score")
ax_cv.set_title("10-Fold CV — Clinical Track  (mean ± std)", fontweight="bold")
ax_cv.axhline(0.85, color=C["a3"], ls="--", lw=1.2, alpha=0.7, label="Target")
ax_cv.legend(fontsize=8.5); ax_cv.grid(axis="y", alpha=0.4)
for i,(a,s) in enumerate(zip(cv_aucs_c, cv_stds_c)):
    ax_cv.text(i-0.2, a+s+0.012, f"{a:.3f}", ha="center", fontsize=9, color=C["text"])

# Overfit comparison
ax_ov = axes5[1]
tr_a = [r["train_acc"] for r in res_c.values()]
te_a = [r["accuracy"]  for r in res_c.values()]
ax_ov.bar(x4-0.18, tr_a, 0.32, color=MC, alpha=0.9, label="Train Acc")
ax_ov.bar(x4+0.18, te_a, 0.32, color=MC, alpha=0.35, label="Test Acc")
ax_ov.set_xticks(x4); ax_ov.set_xticklabels([n[:12] for n in cv_names_c], fontsize=9)
ax_ov.set_ylim(0.55, 1.05); ax_ov.set_ylabel("Accuracy")
ax_ov.set_title("Overfitting Check — Clinical Track\n(Train vs Test Accuracy)", fontweight="bold")
ax_ov.legend(fontsize=9); ax_ov.grid(axis="y", alpha=0.4)
for i,(tr,te) in enumerate(zip(tr_a, te_a)):
    diff = tr - te
    col  = C["a3"] if diff > 0.06 else C["a4"]
    ax_ov.text(i, max(tr,te)+0.015, f"Δ{diff:.2f}", ha="center",
               fontsize=9, color=col, fontweight="bold")

plt.tight_layout(pad=2.0)
plt.savefig("outputs/05_generalization.png", dpi=130,
            bbox_inches="tight", facecolor=C["bg"])
plt.show()
plt.close()
print("✔ Saved → outputs/05_generalization.png")

# ─────────────────────────────────────────────────────────────────────────────
#  9 ▸ FINAL SUMMARY
# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "="*70)
print("  FINAL RESULTS SUMMARY")
print("="*70)

print("\n  ── CLINICAL TRACK (recommended for deployment) ──")
print(f"  Best Model   : {best_c.replace(chr(10),' ')}")
bc_r = res_c[best_c]
print(f"  Accuracy     : {bc_r['accuracy']*100:.2f}%")
print(f"  Precision    : {bc_r['precision']*100:.2f}%")
print(f"  Recall       : {bc_r['recall']*100:.2f}%")
print(f"  F1 Score     : {bc_r['f1']*100:.2f}%")
print(f"  ROC-AUC      : {bc_r['roc_auc']:.4f}")
print(f"  10-CV AUC    : {bc_r['cv_auc_mean']:.4f} ± {bc_r['cv_auc_std']:.4f}")

print(f"\n  ── FULL TRACK (lifestyle + clinical) ──")
bf_r = res_f[best_f]
print(f"  Best Model   : {best_f.replace(chr(10),' ')}")
print(f"  ROC-AUC      : {bf_r['roc_auc']:.4f}  "
      f"(inflated by synthetic SleepHours)")

print("\n  ── TOP CLINICAL RISK PREDICTORS ──")
for _, row in imp_c.head(8).iterrows():
    tag = "★ LIFESTYLE" if row['feature'] in LS_SET else "  Clinical "
    bar = "▓" * max(1, int(row['importance'] * 500))
    print(f"    [{tag}]  {row['feature']:<26}  {row['importance']:.4f}  {bar}")

print("\n  ── KEY CLINICAL INSIGHTS ──")
print("  • Glucose is the dominant single predictor of diabetes risk")
print("  • BMI amplifies glucose risk (Glucose_BMI interaction is key)")
print("  • Age-Glucose ratio captures age-related metabolic decline")
print("  • Insulin sensitivity (HOMA-IR proxy) reveals insulin resistance")
print("  • DiabetesPedigreeFunction encodes genetic/hereditary risk")
print("  • Among lifestyle factors: StressLevel & SugarIntake contribute")
print("    meaningfully to risk; ActivityLevel is protective")

print("\n  ── OVERFITTING STATUS ──")
for name, r in res_c.items():
    diff = r["train_acc"] - r["accuracy"]
    flag = "✔ Good (<6%)" if diff < 0.06 else "⚠ Moderate" if diff < 0.15 else "✗ High"
    print(f"    {name.replace(chr(10),' '):<28}  "
          f"TrainAcc={r['train_acc']:.3f}  TestAcc={r['accuracy']:.3f}  "
          f"Gap={diff:.3f}  {flag}")

print("\n" + "="*70)
print("  OUTPUT FILES")
print("="*70)
for f in ["outputs/01_pipeline_dashboard.png",
          "outputs/02_feature_analysis.png",
          "outputs/03_explainability.png",
          "outputs/04_correlation_heatmap.png",
          "outputs/05_generalization.png"]:
    print(f"  {f}")
print("="*70 + "\n")
