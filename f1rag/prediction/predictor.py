"""Podium prediction (Section 4). Loads the saved Milestone 1 pipeline (model + full preprocessing) from file,
never fits anything, and explains one row live with the same SHAP setup as Milestone 1 (permutation explainer over
the same 100 training background rows).

Pre-race feature values for a known driver-race come from models/prerace_feature_rows.csv, which the Milestone 1
notebook computed from the CSVs with its leakage rules (form and standings from earlier races only). The knowledge
graph is not extended for them.
"""
import json
import sys
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from .. import config
from . import podium_preprocessing

# The pickle refers to the class as podium_preprocessing.PodiumPreprocessor (see the notebook's M3 export cell).
sys.modules.setdefault('podium_preprocessing', podium_preprocessing)

PLAIN = {   # plain-language names used in the form, the plot and the explanation prompt
    'grid_position': 'starting grid position', 'pit_lane_start': 'pit-lane start',
    'field_size': 'number of cars on the grid', 'q1_gap_pct': 'Q1 gap to the fastest lap (%)',
    'q1_gap_to_teammate_pct': 'Q1 gap to the teammate (%)', 'grid_minus_quali': 'places lost from qualifying to grid',
    'driver_podium_rate_last5': "driver's podium rate, last 5 races",
    'driver_avg_finish_rel_last5': "driver's average relative finish, last 5 races",
    'driver_finish_rate_last5': "driver's finishing rate, last 5 races",
    'driver_form_races': 'races behind the driver form figures (0-5)', 'driver_experience': 'driver experience',
    'constructor_podium_rate_last5': "team's podium rate, last 5 races",
    'constructor_best_finish_rel_last5': "team's best relative finish, last 5 races",
    'constructor_finish_rate_last5': "team's finishing rate, last 5 races",
    'constructor_form_races': 'races behind the team form figures (0-5)',
    'driver_champ_rank': "driver's championship position before the race",
    'driver_points_share': "driver's points as a share of the leader's",
    'constructor_champ_rank': "team's championship position before the race",
    'constructor_points_share': "team's points as a share of the leading team's",
    'season_progress': 'share of the season completed', 'standings_from_last_season': "standings from last season",
    'home_race': 'home race'}
UNITS = {'grid_position': 'slot (1 = pole; pit-lane start = field size + 1)', 'field_size': 'cars',
         'q1_gap_pct': '% slower than the fastest Q1 lap', 'q1_gap_to_teammate_pct': '% slower than the teammate',
         'grid_minus_quali': 'places (positive = penalty, negative = gained)',
         'driver_avg_finish_rel_last5': 'finish / field size (0.05 = P1 of 20, 1.0 = last)',
         'constructor_best_finish_rel_last5': 'best finish / field size',
         'driver_experience': 'log(1 + earlier starts)', 'driver_champ_rank': 'position',
         'constructor_champ_rank': 'position', 'season_progress': '0 = opener, 0.5 = halfway'}


class ValidationError(ValueError):
    pass


@dataclass
class Prediction:
    probability: float
    podium: bool
    threshold: float
    row: dict
    contributions: dict = field(default_factory=dict)     # feature -> SHAP value (probability units)
    base_value: float = None
    actual_finish: int = None
    race: dict = None


class PodiumPredictor:
    def __init__(self, models_dir=None):
        import joblib
        d = models_dir or config.MODELS_DIR
        self.model = joblib.load(d / 'podium_model.joblib')
        self.card = json.loads((d / 'model_card.json').read_text(encoding='utf-8'))
        self.features = self.card['features']
        self.threshold = float(self.card['threshold'])
        self.ranges = json.loads((d / 'feature_ranges.json').read_text(encoding='utf-8'))
        self.background = pd.read_csv(d / 'shap_background.csv')[self.features].astype(float)
        self.global_importance = pd.read_csv(d / 'shap_global_importance.csv')
        self.rows = pd.read_csv(d / 'prerace_feature_rows.csv')
        self._lime_rows_path = d / 'lime_training_rows.csv'
        self._explainer = None
        self._parity_check(d)

    def _parity_check(self, d):
        ref = pd.read_csv(d / 'reference_test_predictions.csv').merge(self.rows, on=['raceId', 'driverId'])
        p = self.model.predict_proba(ref[self.features].astype(float))[:, 1]
        err = float(np.abs(p - ref.podium_probability).max())
        if err > 1e-9:
            raise RuntimeError(f'loaded model does not reproduce the notebook test predictions (max diff {err})')
        self.parity_max_diff = err

    # ---------------------------------------------------------------- pre-race rows
    def lookup(self, driver_id, race_id):
        hit = self.rows[(self.rows.driverId == driver_id) & (self.rows.raceId == race_id)]
        return None if hit.empty else hit.iloc[0].to_dict()

    def defaults(self):
        return {f: self.ranges[f]['train_median'] for f in self.features}

    # ---------------------------------------------------------------- validation (manual input)
    def validate(self, values, allowed_missing=()):
        """values: feature -> number or None. Returns a clean dict or raises ValidationError listing every problem.
        A missing value is refused unless the user explicitly marked it as not available (allowed_missing) and the
        feature can legitimately be missing (e.g. a rookie has no form figures)."""
        problems, clean = [], {}
        for f in self.features:
            spec, v = self.ranges[f], values.get(f)
            label = PLAIN[f]
            if v is None or (isinstance(v, float) and np.isnan(v)):
                if f in allowed_missing and spec['missing_allowed']:
                    clean[f] = np.nan
                    continue
                problems.append(f'{label}: a value is required.')
                continue
            try:
                v = float(v)
            except (TypeError, ValueError):
                problems.append(f'{label}: "{v}" is not a number.')
                continue
            if spec['kind'] in ('int', 'binary') and not float(v).is_integer():
                problems.append(f'{label}: must be a whole number.')
            if not spec['min'] <= v <= spec['max']:
                problems.append(f"{label}: {v:g} is outside the valid range {spec['min']:g} to {spec['max']:g}.")
            clean[f] = v
        if not problems:
            g, n = clean['grid_position'], clean['field_size']
            if g > n + 1:
                problems.append(f'starting grid position {g:g} is behind the last slot of a {n:g}-car field '
                                f'(at most {n + 1:g}, used for a pit-lane start).')
            if clean['pit_lane_start'] == 1 and g != n + 1:
                problems.append(f'a pit-lane start must use grid position {n + 1:g} (field size + 1), '
                                f'as in Milestone 1.')
        if problems:
            raise ValidationError(problems)
        return clean

    # ---------------------------------------------------------------- prediction + SHAP
    def predict(self, clean, explain=True, race=None, actual_finish=None):
        X = pd.DataFrame([clean])[self.features].astype(float)
        p = float(self.model.predict_proba(X)[:, 1][0])
        pred = Prediction(round(p, 4), p >= self.threshold, self.threshold, clean, race=race,
                          actual_finish=actual_finish)
        if explain:
            sv = self.explainer(X)
            pred.contributions = dict(zip(self.features, map(float, sv.values[0])))
            pred.base_value = float(np.atleast_1d(sv.base_values)[0])
        return pred

    def explainer(self, X):
        if self._explainer is None:
            import shap
            f = lambda a: self.model.predict_proba(pd.DataFrame(a, columns=self.features))[:, 1]
            self._explainer = shap.Explainer(f, shap.maskers.Independent(self.background),
                                             feature_names=self.features, seed=self.card.get('seed', 42))
        return self._explainer(X)

    def lime(self, clean, num_features=10):
        """Optional LIME view with the Milestone 1 setup (training rows filled with training medians)."""
        import lime.lime_tabular
        train = pd.read_csv(self._lime_rows_path)[self.features].astype(float)
        filled = train.fillna(train.median())
        exp = lime.lime_tabular.LimeTabularExplainer(filled.to_numpy(), feature_names=self.features,
                                                     class_names=['no podium', 'podium'], mode='classification',
                                                     discretize_continuous=True, random_state=self.card.get('seed', 42))
        row = pd.Series(clean)[self.features].astype(float).fillna(train.median()).to_numpy()
        e = exp.explain_instance(row, lambda a: self.model.predict_proba(pd.DataFrame(a, columns=self.features)),
                                 num_features=num_features, num_samples=5000)
        return e.as_list()

    # ---------------------------------------------------------------- explanation context
    def top_contributions(self, pred, k=3):
        items = sorted(pred.contributions.items(), key=lambda kv: kv[1])
        down = [(f, v) for f, v in items if v < 0][:k]
        up = [(f, v) for f, v in reversed(items) if v > 0][:k]
        return up, down

    def explanation_context(self, pred, driver_name, race_name, graph_facts):
        up, down = self.top_contributions(pred)
        fmt = lambda pairs: '\n'.join(f'  - {PLAIN[f]} = {_v(pred.row[f])}: {v * 100:+.1f} percentage points'
                                      for f, v in pairs) or '  (none)'
        top_global = self.global_importance.sort_values('mean_abs_shap', ascending=False).head(5)
        glob = ', '.join(f"{PLAIN.get(r.feature, r.feature)} ({r.mean_abs_shap * 100:.1f} pp average effect)"
                         for r in top_global.itertuples())
        values = '\n'.join(f'  - {PLAIN[f]}: {_v(pred.row[f])}' for f in self.features)
        facts = '\n'.join(f'  - {json.dumps(r, default=str)}' for r in graph_facts) or '  (none available)'
        return (f'Driver and race: {driver_name}, {race_name}\n'
                f'Model output: podium probability {pred.probability:.3f} '
                f'({pred.probability * 100:.1f}%); threshold {pred.threshold:.3f}; predicted class: '
                f"{'podium' if pred.podium else 'not podium'}\n"
                f'Average prediction of the model (SHAP base value): {pred.base_value * 100:.1f}%\n'
                f'Feature values:\n{values}\n'
                f'Strongest contributions pushing the probability UP:\n{fmt(up)}\n'
                f'Strongest contributions pushing the probability DOWN:\n{fmt(down)}\n'
                f'Milestone 1 global SHAP findings (features the model relies on most, on average): {glob}\n'
                f'Pre-race facts from the knowledge graph (races before this one only):\n{facts}')

    def fallback_explanation(self, pred):
        up, down = self.top_contributions(pred)
        s = (f"The model gives a podium probability of {pred.probability * 100:.1f}% "
             f"({'podium' if pred.podium else 'not podium'} at the {pred.threshold * 100:.1f}% threshold). ")
        if up:
            s += 'It relied most on these to push the estimate up: ' + ', '.join(
                f'{PLAIN[f]} ({v * 100:+.1f} pp)' for f, v in up) + '. '
        if down:
            s += 'These pushed it down: ' + ', '.join(f'{PLAIN[f]} ({v * 100:+.1f} pp)' for f, v in down) + '. '
        return s + 'These are the model\'s contributions, not causes of a result.'


def _v(x):
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return 'not available'
    return f'{x:.3g}' if isinstance(x, float) else str(x)
