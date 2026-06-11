from __future__ import annotations
import os, json
import numpy as np
import pandas as pd
from sklearn.metrics import precision_recall_fscore_support, f1_score

KPIS = [
    'downlink_traffic_gb','active_users','call_setup_success_rate_pct','call_drop_rate_pct',
    'handover_success_rate_pct','avg_signal_quality_db','retransmission_rate_pct','latency_ms',
    'cpu_load_pct','unit_temperature_c','supply_voltage_v','uplink_interference_dbm'
]

def intervals_overlap(a0,a1,b0,b1):
    return max(pd.Timestamp(a0),pd.Timestamp(b0)) <= min(pd.Timestamp(a1),pd.Timestamp(b1))

def main(root='.'):
    truth=pd.read_csv(os.path.join(root,'ground_truth','fault_events.csv'), parse_dates=['start_ts','end_ts'])
    truth=truth[truth['window']=='eval'].reset_index(drop=True)
    det=pd.read_csv(os.path.join(root,'data','detected_events.csv'), parse_dates=['start_ts','end_ts'])
    scores=pd.read_csv(os.path.join(root,'data','scores.csv'), parse_dates=['timestamp'])
    tele=pd.read_csv(os.path.join(root,'data','telemetry.csv'), parse_dates=['timestamp'])
    eval_start=tele.timestamp.min()+pd.Timedelta(days=10)
    # Event matching: one detected event matches a truth event if same site and intervals overlap by >= 15 minutes or detection occurs inside fault.
    matched_truth=set(); matched_det=set(); delays=[]; attr_hits=[]; archetype_rows=[]
    for ti,t in truth.iterrows():
        candidates=[]
        for di,d in det.iterrows():
            if d.site_id != t.site_id: continue
            if intervals_overlap(d.start_ts,d.end_ts,t.start_ts,t.end_ts):
                ov=(min(d.end_ts,t.end_ts)-max(d.start_ts,t.start_ts)).total_seconds()/60
                candidates.append((max(ov,0), di, d))
        if candidates:
            candidates.sort(reverse=True, key=lambda x:x[0])
            _,di,d=candidates[0]
            matched_truth.add(ti); matched_det.add(di)
            first_alert=max(pd.Timestamp(d.start_ts), pd.Timestamp(t.start_ts))
            delays.append(max(0,(first_alert-pd.Timestamp(t.start_ts)).total_seconds()/60))
            truth_kpis=set(str(t.affected_kpis).split(';'))
            pred_top={d.top1_kpi,d.top2_kpi,d.top3_kpi}
            attr_hits.append(int(len(truth_kpis & pred_top)>0))
            archetype_rows.append({'fault_type':t.fault_type,'detected':1,'delay_min':delays[-1],
                                  'attr_hit':attr_hits[-1],'predicted_fault_type':d.predicted_fault_type})
        else:
            archetype_rows.append({'fault_type':t.fault_type,'detected':0,'delay_min':np.nan,'attr_hit':0,'predicted_fault_type':''})
    tp=len(matched_truth); fp=len(det)-len(matched_det); fn=len(truth)-tp
    event_precision=tp/(tp+fp) if tp+fp else 0
    event_recall=tp/(tp+fn) if tp+fn else 0
    event_f1=2*event_precision*event_recall/(event_precision+event_recall) if event_precision+event_recall else 0
    # Point labels for eval window only
    eval_scores=scores[scores.timestamp>=eval_start].copy()
    eval_scores['y_true']=0
    for _,t in truth.iterrows():
        m=(eval_scores.site_id==t.site_id)&(eval_scores.timestamp>=t.start_ts)&(eval_scores.timestamp<=t.end_ts)
        eval_scores.loc[m,'y_true']=1
    y_true=eval_scores.y_true.values; y_pred=eval_scores.alert.values
    p,r,f,_=precision_recall_fscore_support(y_true,y_pred,average='binary',zero_division=0)
    # point-adjust: if any point in a true event is detected, mark all points in that event as detected
    y_pa=y_pred.copy()
    for _,t in truth.iterrows():
        m=((eval_scores.site_id==t.site_id)&(eval_scores.timestamp>=t.start_ts)&(eval_scores.timestamp<=t.end_ts)).values
        if np.any(y_pred[m]==1): y_pa[m]=1
    ppa,rpa,fpa,_=precision_recall_fscore_support(y_true,y_pa,average='binary',zero_division=0)
    days=(eval_scores.timestamp.max()-eval_scores.timestamp.min()).total_seconds()/86400 + 15/1440
    false_alert_points=((eval_scores.alert==1)&(eval_scores.y_true==0)).sum()
    false_alarms_per_site_day=false_alert_points/(eval_scores.site_id.nunique()*days)
    arch=pd.DataFrame(archetype_rows)
    per_arch=arch.groupby('fault_type').agg(events=('detected','count'),recall=('detected','mean'),mean_delay_min=('delay_min','mean'),attr_hit_rate=('attr_hit','mean')).reset_index()
    metrics={
        'n_truth_eval_events': int(len(truth)), 'n_detected_events': int(len(det)),
        'event_precision': event_precision, 'event_recall': event_recall, 'event_f1': event_f1,
        'point_precision': float(p), 'point_recall': float(r), 'point_f1': float(f),
        'point_adjust_precision': float(ppa), 'point_adjust_recall': float(rpa), 'point_adjust_f1': float(fpa),
        'mean_detection_delay_min': float(np.nanmean(delays)) if delays else None,
        'false_alert_points_per_site_per_day': float(false_alarms_per_site_day),
        'attribution_hit_rate': float(np.mean(attr_hits)) if attr_hits else 0.0,
    }
    os.makedirs(os.path.join(root,'data'), exist_ok=True)
    with open(os.path.join(root,'data','metrics.json'),'w') as f: json.dump(metrics,f,indent=2)
    per_arch.to_csv(os.path.join(root,'data','per_archetype_metrics.csv'), index=False)
    eval_scores[['site_id','timestamp','alert','y_true']].to_csv(os.path.join(root,'data','point_eval_labels.csv'), index=False)
    print(json.dumps(metrics, indent=2))

if __name__ == '__main__':
    main(root=os.path.abspath(os.path.join(os.path.dirname(__file__),'..')))
