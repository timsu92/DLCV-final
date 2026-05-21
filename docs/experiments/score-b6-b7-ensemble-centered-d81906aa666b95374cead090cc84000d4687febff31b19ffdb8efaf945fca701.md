# score: b6-b7-ensemble-centered

Recorded at: 2026-05-20T19:36:31

Same B6+B7 models, ensemble re-run with training-mean centering of embed_features before KNN. new_ratio=0.165, threshold=0.3334 (was 0.5118 raw). rows=27956, new_individual_ratio=0.057. Fix: subtract per-model-pair training feature mean from both train and test features before KNN search. This removed the common background direction shared by all whale images (BatchNorm1d neck collapse). Intra/inter cosine gap: 0.028→0.508 (18x). Run dir: results/2026-05-20-192514-knshnb-b6-b7-ensemble-centered

However, the score raised only from 0.07149 to 0.13169