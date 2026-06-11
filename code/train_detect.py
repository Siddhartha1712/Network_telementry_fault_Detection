from __future__ import annotations
import os, json
import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import RobustScaler
from sklearn.decomposition import PCA

KPIS = [
    'downlink_traffic_gb','active_users','call_setup_success_rate_pct','call_drop_rate_pct',
    'handover_success_rate_pct','avg_signal_quality_db','retransmission_rate_pct','latency_ms',
    'cpu_load_pct','unit_temperature_c','supply_voltage_v','uplink_interference_dbm'
]
FAULT_RULES = {
    'F1': {'pos':['call_drop_rate_pct','retransmission_rate_pct','latency_ms'], 'neg':['avg_signal_quality_db']},
    'F2': {'pos':['call_drop_rate_pct'], 'neg':['handover_success_rate_pct']},
    'F3': {'pos':['unit_temperature_c','cpu_load_pct','latency_ms'], 'neg':[], 'special':['supply_voltage_v']},
    'F4': {'pos':[], 'neg':['downlink_traffic_gb','active_users']},
    'F5': {'pos':['uplink_interference_dbm','retransmission_rate_pct','latency_ms'], 'neg':['avg_signal_quality_db','call_setup_success_rate_pct','handover_success_rate_pct']},
    'F6': {'pos':['downlink_traffic_gb','active_users','latency_ms','cpu_load_pct','call_drop_rate_pct'], 'neg':['call_setup_success_rate_pct']},
}

def add_time_features(df):
    ts=pd.to_datetime(df['timestamp'])
    hour=ts.dt.hour + ts.dt.minute/60
    dow=ts.dt.dayofweek
    out=df.copy()
    out['hour_sin']=np.sin(2*np.pi*hour/24); out['hour_cos']=np.cos(2*np.pi*hour/24)
    out['dow_sin']=np.sin(2*np.pi*dow/7); out['dow_cos']=np.cos(2*np.pi*dow/7)
    return out

def impute_per_site(df):
    out=df.sort_values(['site_id','timestamp']).copy()
    out[KPIS]=out.groupby('site_id')[KPIS].transform(lambda g: g.interpolate(limit_direction='both').ffill().bfill())
    return out

def rolling_context(df):
    out=df.copy()
    for col in KPIS:
        # Use causal rolling median by site to remove daily level shifts but keep anomalies
        med=out.groupby('site_id')[col].transform(lambda s: s.rolling(96, min_periods=8).median())
        site_med=out.groupby('site_id')[col].transform('median')
        med=med.fillna(site_med)
        out[col+'_dev']=(out[col]-med).fillna(0)
    return out

def eventize(scores, min_len=2, gap=1):
    events=[]
    for site, g in scores.groupby('site_id'):
        g=g.sort_values('timestamp').reset_index(drop=True)
        in_ev=False; start=None; end=None; count=0; gap_count=0
        for _,r in g.iterrows():
            if r['alert']==1:
                if not in_ev:
                    in_ev=True; start=r['timestamp']; count=0
                count+=1; end=r['timestamp']; gap_count=0
            else:
                if in_ev:
                    gap_count+=1
                    if gap_count>gap:
                        if count>=min_len: events.append({'site_id':site,'start_ts':start,'end_ts':end})
                        in_ev=False; start=None; end=None; count=0; gap_count=0
        if in_ev and count>=min_len: events.append({'site_id':site,'start_ts':start,'end_ts':end})
    return pd.DataFrame(events)

def predict_fault_and_topk(feature_deviation):
    
    # feature_deviation is mean signed robust deviation per KPI for an event
    absdev=feature_deviation.abs().sort_values(ascending=False)
    top3=list(absdev.head(3).index)
    scores={}
    for ft,rule in FAULT_RULES.items():
        sc=0.0
        for k in rule.get('pos',[]): sc += max(feature_deviation.get(k,0),0)
        for k in rule.get('neg',[]): sc += max(-feature_deviation.get(k,0),0)
        for k in rule.get('special',[]): sc += abs(feature_deviation.get(k,0))
        scores[ft]=float(sc)
    pred=max(scores, key=scores.get)
    return top3, pred, scores

def main(root='.'):
    data_path=os.path.join(root,'data','telemetry.csv')
    df=pd.read_csv(data_path, parse_dates=['timestamp'])
    df=impute_per_site(df)
    df=add_time_features(df)
    df=rolling_context(df)
    train_end=df['timestamp'].min()+pd.Timedelta(days=10)
    train=df[df.timestamp < train_end].copy()
    feats=KPIS + [c+'_dev' for c in KPIS] + ['hour_sin','hour_cos','dow_sin','dow_cos']


    # Fit only on training window; no ground-truth folder is imported here.
    scaler=RobustScaler()
    Xtr=scaler.fit_transform(train[feats])
    X=scaler.transform(df[feats])
    iso=IsolationForest(n_estimators=220, max_samples=0.7, contamination=0.025, random_state=7, n_jobs=-1)
    iso.fit(Xtr)
    iso_score=-iso.score_samples(X)

    # PCA reconstruction residual as a complementary smoother detector
    pca=PCA(n_components=0.88, random_state=7)
    pca.fit(Xtr)
    Xhat=pca.inverse_transform(pca.transform(X))
    resid=np.mean((X-Xhat)**2, axis=1)

    # Robust z baseline score
    med=np.nanmedian(Xtr, axis=0); mad=np.nanmedian(np.abs(Xtr-med), axis=0)+1e-6
    z=np.max(np.abs((X-med)/mad), axis=1)
    def rank01(a):
        return pd.Series(a).rank(pct=True).values
    df['score_iso']=rank01(iso_score); df['score_pca']=rank01(resid); df['score_z_baseline']=rank01(z)
    df['anomaly_score']=0.50*df['score_iso'] + 0.30*df['score_pca'] + 0.20*df['score_z_baseline']


    # Threshold: choose per-site threshold from training scores to meet <=2 false alerts/site/day.
    # 10 days x 96 points/day = 960 points. 2/day => about 20 alert points/site in training.
    thresholds={}
    for site,g in df[df.timestamp < train_end].groupby('site_id'):
        q=1 - min(12/len(g), 0.04)
        thresholds[site]=float(g['anomaly_score'].quantile(q))
    df['threshold']=df['site_id'].map(thresholds)
    df['alert']=(df['anomaly_score'] >= df['threshold']).astype(int)


    # Save scores only for eval window for normal submission; train scores retained for reproducibility.
    score_cols=['site_id','timestamp','anomaly_score','score_iso','score_pca','score_z_baseline','threshold','alert']
    df[score_cols].to_csv(os.path.join(root,'data','scores.csv'), index=False)


    # Event attribution using deviations from training medians by site and KPI.
    train_meds=train.groupby('site_id')[KPIS].median()
    train_mads=(train.groupby('site_id')[KPIS].apply(lambda x: (x-x.median()).abs().median())+1e-6)
    events=eventize(df[df.timestamp>=train_end][['site_id','timestamp','alert']], min_len=2, gap=1)
    rows=[]
    for i,ev in events.iterrows():
        mask=(df.site_id==ev.site_id)&(df.timestamp>=ev.start_ts)&(df.timestamp<=ev.end_ts)
        site=ev.site_id
        dev=((df.loc[mask, KPIS].mean()-train_meds.loc[site])/train_mads.loc[site]).replace([np.inf,-np.inf],0).fillna(0)
        top3,pred,rule_scores=predict_fault_and_topk(dev)
        rows.append({'detected_event_id':f'D{i+1:03d}','site_id':site,'start_ts':ev.start_ts,'end_ts':ev.end_ts,
                     'top1_kpi':top3[0] if len(top3)>0 else '', 'top2_kpi':top3[1] if len(top3)>1 else '', 'top3_kpi':top3[2] if len(top3)>2 else '',
                     'predicted_fault_type':pred, 'rule_scores_json':json.dumps(rule_scores)})
    pd.DataFrame(rows).to_csv(os.path.join(root,'data','detected_events.csv'), index=False)
    with open(os.path.join(root,'data','thresholds.json'),'w') as f: json.dump(thresholds,f,indent=2)
    print(f"Detected {len(rows)} events. Scores written to data/scores.csv")

if __name__ == '__main__':
    main(root=os.path.abspath(os.path.join(os.path.dirname(__file__),'..')))
