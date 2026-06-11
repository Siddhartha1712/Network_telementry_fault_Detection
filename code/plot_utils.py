from __future__ import annotations
import os, json
import pandas as pd
import matplotlib.pyplot as plt

PLOT_KPIS=['downlink_traffic_gb','active_users','handover_success_rate_pct','avg_signal_quality_db','supply_voltage_v','uplink_interference_dbm']

def main(root='.'):
    os.makedirs(os.path.join(root,'plots'), exist_ok=True)
    tele=pd.read_csv(os.path.join(root,'data','telemetry.csv'), parse_dates=['timestamp'])
    scores=pd.read_csv(os.path.join(root,'data','scores.csv'), parse_dates=['timestamp'])
    truth=pd.read_csv(os.path.join(root,'ground_truth','fault_events.csv'), parse_dates=['start_ts','end_ts'])
    metrics=json.load(open(os.path.join(root,'data','metrics.json')))
    # site profile plot
    fig, ax = plt.subplots(figsize=(9,4))
    for site,g in tele.groupby('site_id'):
        daily=g.set_index('timestamp')['downlink_traffic_gb'].resample('D').mean()
        ax.plot(daily.index, daily.values, alpha=.75, label=site)
    ax.set_title('Daily mean downlink traffic by site')
    ax.set_ylabel('GB / 15-min interval')
    ax.legend(ncol=5, fontsize=6)
    fig.autofmt_xdate(); fig.tight_layout(); fig.savefig(os.path.join(root,'plots','site_traffic_profiles.png'), dpi=160); plt.close(fig)
    # score plot with true event windows for a site that has an F2 if possible
    eval_truth=truth[truth.window=='eval']
    row=eval_truth[eval_truth.fault_type=='F2'].iloc[0] if len(eval_truth[eval_truth.fault_type=='F2']) else eval_truth.iloc[0]
    site=row.site_id
    g=scores[scores.site_id==site]
    fig, ax=plt.subplots(figsize=(9,3.5))
    ax.plot(g.timestamp, g.anomaly_score, label='anomaly score')
    ax.plot(g.timestamp, g.threshold, linestyle='--', label='threshold')
    for _,t in eval_truth[eval_truth.site_id==site].iterrows():
        ax.axvspan(t.start_ts,t.end_ts, alpha=.2)
        ax.text(t.start_ts, ax.get_ylim()[1]*.92, t.fault_type, fontsize=8)
    ax.set_title(f'Anomaly score and true fault windows - {site}')
    ax.legend(); fig.autofmt_xdate(); fig.tight_layout(); fig.savefig(os.path.join(root,'plots','score_case_study.png'), dpi=160); plt.close(fig)
    # per archetype recall
    pa=pd.read_csv(os.path.join(root,'data','per_archetype_metrics.csv'))
    fig, ax=plt.subplots(figsize=(6,3.5))
    ax.bar(pa.fault_type, pa.recall)
    ax.set_ylim(0,1); ax.set_ylabel('Event recall'); ax.set_title('Recall by fault archetype')
    fig.tight_layout(); fig.savefig(os.path.join(root,'plots','per_archetype_recall.png'), dpi=160); plt.close(fig)
    # missingness
    miss=tele.isna().mean().sort_values(ascending=False).head(12)
    fig, ax=plt.subplots(figsize=(7,3.5))
    ax.bar(miss.index, miss.values*100); ax.set_ylabel('% missing'); ax.set_title('Benign missingness rate by column')
    ax.tick_params(axis='x', rotation=55); fig.tight_layout(); fig.savefig(os.path.join(root,'plots','missingness.png'), dpi=160); plt.close(fig)
    print('Plots written to plots/')

if __name__ == '__main__':
    main(root=os.path.abspath(os.path.join(os.path.dirname(__file__),'..')))
