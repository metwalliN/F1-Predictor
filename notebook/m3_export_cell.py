# M3. Export the frozen final model and everything the Milestone 3 app needs.
# Nothing is fitted here: the objects below are the ones evaluated above (FINAL_MODEL, FINAL_THRESHOLD, the SHAP
# background and the LIME training rows). The app only LOADS these files; it never refits.
import joblib, types, shutil, hashlib

M3_DIR = RUN_DIR / 'm3_artifacts'
M3_DIR.mkdir(exist_ok=True)

# ---- 1. The model with its full preprocessing (one sklearn Pipeline) ----
# PodiumPreprocessor was defined in this notebook (module __main__). Pickle stores classes by module path, so it is
# registered under the module name 'podium_preprocessing'. The repo ships prediction/podium_preprocessing.py with the
# identical class, so the app can unpickle it without this notebook.
_mod = types.ModuleType('podium_preprocessing')
for _name, _val in {'PodiumPreprocessor': PodiumPreprocessor, 'LOG_FEATURE': LOG_FEATURE, 'CLIP_FEATURE': CLIP_FEATURE,
                    'CLIP_QUANTILES': CLIP_QUANTILES, 'MISSING_FLAGS': MISSING_FLAGS}.items():
    setattr(_mod, _name, _val)
sys.modules['podium_preprocessing'] = _mod
PodiumPreprocessor.__module__ = 'podium_preprocessing'
joblib.dump(FINAL_MODEL, M3_DIR / 'podium_model.joblib')

model_card = {'family': FINAL_FAMILY, 'attempt': str(winners.loc[FINAL_FAMILY, 'attempt']),
              'threshold': float(FINAL_THRESHOLD), 'threshold_rule': 'maximises F1 on validation (2020-2021)',
              'features': FEATURES, 'seed': SEED, 'train_seasons': SPLITS['train'], 'model_start_year': MODEL_START_YEAR,
              'sklearn_version': importlib.metadata.version('scikit-learn'),
              'shap_setup': 'shap.Explainer(predict_proba[:,1], shap.maskers.Independent(background), feature_names=FEATURES); '
                            'background = X_train.sample(100, random_state=SEED)',
              'lime_setup': 'LimeTabularExplainer(X_train filled with train medians, discretize_continuous=True, '
                            'random_state=SEED), num_samples=5000'}
(M3_DIR / 'model_card.json').write_text(json.dumps(model_card, indent=2, default=str))

# ---- 2. Explainer inputs, exactly as used in X2/X3 ----
background.to_csv(M3_DIR / 'shap_background.csv', index=False)                 # the same 100 training rows
X['train'].to_csv(M3_DIR / 'lime_training_rows.csv', index=False)               # LIME fills these with train medians
compare.reset_index().rename(columns={'index': 'feature'}).to_csv(M3_DIR / 'shap_global_importance.csv', index=False)

# ---- 3. Pre-race feature rows for every modeled driver-race (computed above with the FE1-FE4 leakage rules) ----
names = (analysis_base[['raceId', 'driverId', 'race_name', 'driver_forename', 'driver_surname', 'positionOrder']]
         .astype({'raceId': 'int64', 'driverId': 'int64'}))
cons = clean['constructors'][['constructorId', 'name']].rename(columns={'name': 'constructor_name'}).astype({'constructorId': 'int64'})
circ = clean['circuits'][['circuitId', 'name']].rename(columns={'name': 'circuit_name'}).astype({'circuitId': 'int64'})
rows = (modeling_table.astype({'raceId': 'int64', 'driverId': 'int64', 'constructorId': 'int64', 'circuitId': 'int64'})
        .merge(names, on=['raceId', 'driverId'], how='left', validate='one_to_one')
        .merge(cons, on='constructorId', how='left', validate='many_to_one')
        .merge(circ, on='circuitId', how='left', validate='many_to_one')
        .rename(columns={'positionOrder': 'actual_finish'}))
assert len(rows) == len(modeling_table) and rows.race_name.notna().all() and rows.driver_surname.notna().all()
rows.to_csv(M3_DIR / 'prerace_feature_rows.csv', index=False)

# ---- 4. Valid input ranges for the manual form (observed over the whole 2003-2024 modeling table) ----
register = pd.read_csv(CLEAN_DIR / 'feature_register.csv').drop_duplicates('feature', keep='last').set_index('feature')
mt = modeling_table[FEATURES].astype(float)
train_rows = modeling_table.split.eq('train')
BOUNDS = {f: (0.0, 1.0) for f in FEATURES if f.endswith(('_rate_last5', '_points_share', '_rel_last5')) or f == 'season_progress'}
ranges = {}
for f in FEATURES:
    v = mt[f].dropna()
    is_int = bool(np.allclose(v, v.round()))
    is_binary = set(v.unique()) <= {0.0, 1.0}
    lo, hi = BOUNDS.get(f, (float(v.min()), float(v.max())))
    ranges[f] = {'group': register.loc[f, 'group'], 'description': register.loc[f, 'pre_race_reason'],
                 'kind': 'binary' if is_binary else 'int' if is_int else 'float',
                 'min': lo, 'max': hi, 'observed_min': float(v.min()), 'observed_max': float(v.max()),
                 'p01': float(v.quantile(0.01)), 'p99': float(v.quantile(0.99)),
                 'train_median': float(modeling_table.loc[train_rows, f].median()),
                 'missing_allowed': f in WHY_MISSING, 'why_missing': WHY_MISSING.get(f, '')}
(M3_DIR / 'feature_ranges.json').write_text(json.dumps(ranges, indent=2))

# ---- 5. Parity reference: the app must reproduce these test probabilities exactly ----
ref = part['test'][KEY].copy()
ref['podium_probability'] = PROBS[FINAL_FAMILY]['test']
ref.to_csv(M3_DIR / 'reference_test_predictions.csv', index=False)
shutil.copy(CLEAN_DIR / 'feature_spec.json', M3_DIR / 'feature_spec.json')

for p in sorted(M3_DIR.iterdir()):
    print(f'{p.name:<34} {p.stat().st_size/1024:>9.1f} KB  sha256 {hashlib.sha256(p.read_bytes()).hexdigest()[:12]}')
print(f'Copy the folder {M3_DIR} into the repo as models/ (Kaggle: Output tab -> download).')
