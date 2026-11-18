# PathSnap: First Evaluation (EDA, Feature Selection, Model Comparison)

UE24CS352A Machine Learning Hackathon, PES University, 18 Nov 2026.
The task: match noisy GPS points to the road segment (arc) the vehicle was actually on.

| Notebook | Contents |
|---|---|
| [01_eda.ipynb](notebooks/01_eda.ipynb) | Dataset inventory, missing values, sampling gaps, road-network connectivity, GPS noise, Folium maps |
| [02_feature_selection.ipynb](notebooks/02_feature_selection.ipynb) | Feature distributions, correlation, mutual information, RF and permutation importance, drop-one-feature ablation |
| [03_model_comparison.ipynb](notebooks/03_model_comparison.ipynb) | 6 algorithms: CV and student-test metrics, confusion matrices, MLP training curve |

Partitions: **train** (80 trips) is used for fitting, cross-validation and every selection decision. **student_test** (10 trips) is used for comparison only. Evaluator data is never used.

## 1. EDA: key findings
- 80 train and 10 student-test trips. Each trip has its own road network (`.nodes`, `.arcs`), a GPS track (`.track`) and a reference route (`.route`). The route is an **ordered list of arcs**, not one label per GPS point.
- **0** missing values and **0** non-increasing timestamps. **96.5%** of steps are exactly 1 s.
- **69 gaps longer than 60 s** in train. The worst is trip 90 at 153,064 s (about 42 h).
- Trips span **25 UTM zones in both hemispheres**, so each trip is projected into its own UTM zone (metres, not degrees).
- GPS noise: the median distance to the reference route is **5.4 m** and the 90th percentile is 19.3 m. That justifies a 50 m candidate radius, which misses the true arc for only **0.4%** of student-test points.
- Road networks have 0 isolated nodes. The largest connected component holds about 97% of nodes, and about 87% of arcs have a reverse twin (two-way streets).
- Label anomalies: in train trip 54 the GPS lies off the network; in train trip 33 the route is offset about 120 m. Unusable points are excluded from training.

## 2. Problem setup
For each cleaned GPS point (3 m stationary filter, 60 s gap split), every arc within 50 m (at most 8) is a **candidate**. Each (point, candidate) pair is one row, labelled 1 if it is the point's true arc. The true arc is the nearest arc on the reference route; ties between the two directions of a street are broken by the direction of travel.
This gives 331,631 training rows (usable points, every 3rd point), of which 17.6% are positive.

## 3. Feature selection
| Feature | Meaning | Mutual info | Permutation F1 drop | CV F1 change if removed |
|---|---|---|---|---|
| heading_diff_deg | GPS direction vs arc direction | 0.097 | **0.185** | **−0.131** |
| dist_minus_min | extra distance vs closest candidate | 0.174 | 0.097 | −0.008 |
| frac_along | position of projection on the arc | 0.115 | 0.018 | −0.008 |
| arc_len_m | arc length | 0.064 | 0.004 | −0.004 |
| speed_mps | GPS speed | 0.000 | 0.005 | −0.003 |
| dist_m | perpendicular distance to arc | 0.106 | 0.005 | −0.001 |
| n_candidates | candidates around the point | 0.040 | 0.001 | −0.001 |
| dist_rank | 0 = closest candidate | 0.159 | −0.007 | −0.000 |

Sources: `experiments/feature_importance.csv` and `experiments/feature_ablation.csv`. The ablation is 5-fold GroupKFold CV macro F1 with LightGBM, using all 8 features as the baseline (0.877).
- **heading_diff_deg is essential.** It is the only feature that separates the two directions of a street.
- **speed_mps** carries no information on its own, but it helps in combination with heading, because heading is unreliable at low speed.
- **dist_rank** is redundant with the other distance features: removing it changes nothing. We keep it, but it could be dropped.

## 4. Model comparison (6 algorithms)
| Model | Family | CV macro F1 (5-fold, by trip) | **Accuracy (Top-1)** | Test macro F1 | Log loss | Route continuity | Inference (ms / 1k rows) |
|---|---|---|---|---|---|---|---|
| Logistic Regression | Linear | 0.837 ± 0.010 | 0.840 | 0.844 | 0.244 | 0.911 | 0.05 |
| Gaussian Naive Bayes | Probabilistic | 0.828 ± 0.010 | 0.832 | 0.823 | 0.487 | 0.896 | 0.20 |
| KNN (k=25, distance) | Instance-based | 0.887 ± 0.013 | 0.843 | 0.904 | 0.182 | 0.914 | 22.86 |
| Random Forest (200 trees) | Bagging | 0.880 ± 0.015 | 0.850 | 0.884 | 0.172 | 0.916 | 2.70 |
| LightGBM (300 trees) | Boosting | 0.877 ± 0.016 | **0.856** | 0.879 | 0.177 | 0.920 | 2.77 |
| **MLP (64-32)** | Neural network | **0.892 ± 0.013** | 0.849 | **0.905** | **0.129** | 0.917 | 0.38 |

Test metrics are on the 10 student-test trips, covering all 18,426 cleaned points. **Accuracy** is observation-level Top-1: the highest-probability candidate per GPS point is compared with its true arc, and points whose true arc is missing from the candidates count as wrong.

**Selected: the MLP.** It has the best CV macro F1, which is the primary metric because the classes are imbalanced. It also has the lowest log loss, i.e. the best-calibrated probabilities, which the next stage (HMM/Viterbi) uses directly. Top-1 accuracy differs by under 1 point across RF, LightGBM and MLP. LogReg and NB underfit. KNN is about 60× slower at inference.

## Reproduce
```bash
python3.12 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
python -m src.data.inventory            # dataset inventory
python -m src.features.build_dataset    # candidates, labels, features (~15 s)
python -m src.models.train_compare      # 6 models, CV + student-test (~6 min)
python -m pytest -q tests
```
Then open the notebooks in order. Next stages, built in the main project repo: Bayesian tuning, HMM/Viterbi decoding, then the FastAPI and Streamlit deployment.
