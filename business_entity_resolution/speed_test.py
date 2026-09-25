"""Quick feature speed test"""
import sys, io, time
sys.path.insert(0, '.')

import numpy as np, pandas as pd
from business_entity_resolution.src import config as cfg
from business_entity_resolution.src.data_loader import parse_ground_truth
from business_entity_resolution.src.normalization import normalize_dataframe
from business_entity_resolution.src.blocking import generate_candidates
from business_entity_resolution.src.features import build_feature_matrix, fit_tfidf_computers

print('Loading data...')
s2 = pd.read_csv(cfg.TRAIN_SOURCE2, sep='\t', dtype=str, nrows=2000).fillna('')
s3 = pd.read_csv(cfg.TRAIN_SOURCE3, sep='\t', dtype=str, nrows=2000).fillna('')
target_ids = set(s2['entity_id']) | set(s3['entity_id'])

gt_df = pd.read_csv(cfg.TRAIN_GT, sep='\t', dtype=str, nrows=5000).fillna('')
gt_dict = parse_ground_truth(gt_df)

valid_s1 = [k for k, v in gt_dict.items() if not v or bool(v & target_ids)][:100]
s1_full = pd.read_csv(cfg.TRAIN_SOURCE1, sep='\t', dtype=str).fillna('')
s1 = s1_full[s1_full['entity_id'].isin(set(valid_s1))].reset_index(drop=True)
gt = {k: v for k, v in gt_dict.items() if k in set(valid_s1)}

print(f'S1={len(s1)}, S2={len(s2)}, S3={len(s3)}')

s1 = normalize_dataframe(s1)
s2 = normalize_dataframe(s2)
s3 = normalize_dataframe(s3)

all_names = s1['norm_name'].tolist() + s2['norm_name'].tolist() + s3['norm_name'].tolist()
all_addrs = s1['norm_address'].tolist() + s2['norm_address'].tolist() + s3['norm_address'].tolist()
name_tfidf, addr_tfidf = fit_tfidf_computers(all_names, all_addrs)

print('Running blocking...')
cands = generate_candidates(s1, s2, s3)
pairs = [(s1_id, tgt) for s1_id, cset in cands.items() for tgt in cset]
print(f'Candidate pairs: {len(pairs):,}')

eid = 'entity_id'
s1_dict = {row[eid]: row for _, row in s1.iterrows()}
t_dict = {}
for _, row in s2.iterrows():
    t_dict[row[eid]] = row
for _, row in s3.iterrows():
    t_dict[row[eid]] = row

print('Computing features (optimized)...')
t0 = time.time()
feat_df, feat_names = build_feature_matrix(pairs, s1_dict, t_dict, name_tfidf, addr_tfidf)
elapsed = time.time() - t0
rate = len(pairs) / elapsed if elapsed > 0 else 0
print(f'DONE: {len(pairs):,} pairs, {len(feat_names)} features in {elapsed:.1f}s ({rate:.0f} pairs/sec)')
print(f'Features: {feat_names}')
print(f'DataFrame shape: {feat_df.shape}')
