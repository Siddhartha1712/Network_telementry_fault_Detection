from __future__ import annotations
import os, json, yaml
from dataclasses import dataclass
import numpy as np
import pandas as pd

KPIS = [
    'downlink_traffic_gb','active_users','call_setup_success_rate_pct','call_drop_rate_pct',
    'handover_success_rate_pct','avg_signal_quality_db','retransmission_rate_pct','latency_ms',
    'cpu_load_pct','unit_temperature_c','supply_voltage_v','uplink_interference_dbm'
]
FAULT_KPIS = {
    'F1': ['avg_signal_quality_db','call_drop_rate_pct','retransmission_rate_pct','latency_ms'],
    'F2': ['handover_success_rate_pct','call_drop_rate_pct'],
    'F3': ['supply_voltage_v','unit_temperature_c','cpu_load_pct','latency_ms'],
    'F4': ['downlink_traffic_gb','active_users'],
    'F5': ['uplink_interference_dbm','avg_signal_quality_db','call_setup_success_rate_pct','handover_success_rate_pct','retransmission_rate_pct','latency_ms'],
    'F6': ['downlink_traffic_gb','active_users','latency_ms','cpu_load_pct','call_drop_rate_pct','call_setup_success_rate_pct']
}

def load_config(path: str):
    with open(path, 'r') as f:
        return yaml.safe_load(f)

def daily_shape(hours):
    # busy afternoon/evening, quiet night
    return 0.38 + 0.62 * (np.sin((hours - 6) / 24 * 2*np.pi) + 1) / 2

def weekend_factor(idx):
    dow = idx.dayofweek
    return np.where(dow >= 5, 0.78, 1.0)

def create_base(cfg):
    rng = np.random.default_rng(cfg['seed'])
    start = pd.Timestamp(cfg['start_ts'])
    periods = int(cfg['days'] * 24 * 60 / cfg['freq_minutes'])
    ts = pd.date_range(start, periods=periods, freq=f"{cfg['freq_minutes']}min")
    rows=[]
    site_types = ['urban','urban','suburban','suburban','suburban','rural','rural','rural','highway','industrial']
    base_mult = {'urban': 1.55, 'suburban': 1.0, 'rural': 0.45, 'highway': 0.75, 'industrial': 1.15}
    for s in range(cfg['n_sites']):
        site_id=f"site_{s+1:02d}"
        stype=site_types[s % len(site_types)]
        mult=base_mult[stype] * rng.uniform(0.85,1.2)
        h=ts.hour + ts.minute/60
        season=daily_shape(h) * weekend_factor(ts)
        trend=np.linspace(0, rng.uniform(-0.05,0.06), len(ts))
        traffic_mean=(1.2 + 9.5*season)*mult*(1+trend)
        hetero_noise = rng.normal(0, 0.10 + 0.05*season, len(ts))
        traffic=np.maximum(0.05, traffic_mean*(1+hetero_noise))
        users=np.maximum(1, traffic*rng.uniform(23,38) + rng.normal(0, 6+4*season, len(ts)))
        interference=-112 + 5.5*season + rng.normal(0, 1.2+0.6*season, len(ts)) + (mult-1)*1.2
        quality=-72 - 0.38*(interference+105) - 0.035*users + rng.normal(0,1.0,len(ts))
        latency=24 + 0.05*users + 0.9*np.maximum(interference+108,0) + rng.gamma(1.3, 1.4, len(ts))
        cpu=np.clip(18 + 0.21*users + rng.normal(0, 3.5+1.5*season, len(ts)), 3, 95)
        temp=27 + 0.12*cpu + 3.5*daily_shape(h-2) + rng.normal(0, 1.0, len(ts))
        voltage=48.0 + rng.normal(0, 0.06, len(ts))
        setup=np.clip(99.0 - 0.012*users - 0.22*np.maximum(interference+108,0) - 0.035*np.maximum(latency-35,0) + rng.normal(0,0.28,len(ts)), 90, 99.95)
        handover=np.clip(98.5 - 0.18*np.maximum(interference+108,0) - 0.015*users + rng.normal(0,0.35,len(ts)), 88, 99.8)
        drop=np.clip(0.25 + 0.012*users + 0.16*np.maximum(interference+108,0) + rng.normal(0,0.16+0.1*season,len(ts)), 0.02, 8)
        retx=np.clip(0.5 + 0.20*np.maximum(interference+108,0) + 0.018*users + rng.normal(0,0.25,len(ts)), 0.02, 12)
        df=pd.DataFrame({'site_id':site_id,'site_type':stype,'timestamp':ts,
            'downlink_traffic_gb':traffic,'active_users':users,'call_setup_success_rate_pct':setup,
            'call_drop_rate_pct':drop,'handover_success_rate_pct':handover,'avg_signal_quality_db':quality,
            'retransmission_rate_pct':retx,'latency_ms':latency,'cpu_load_pct':cpu,
            'unit_temperature_c':temp,'supply_voltage_v':voltage,'uplink_interference_dbm':interference})
        rows.append(df)
    return pd.concat(rows, ignore_index=True)

def random_fault_schedule(cfg):
    rng=np.random.default_rng(cfg['seed']+100)
    fault_types=['F1','F2','F3','F4','F5','F6']
    schedule=[]
    eval_counts={ft:0 for ft in fault_types}
    # exactly 3 faint train faults
    for i,ft in enumerate(['F1','F2','F5']):
        site=f"site_{rng.integers(1,cfg['n_sites']+1):02d}"
        start_day=int(rng.integers(2,10)); dur={'F1':8,'F2':14,'F5':1.0}[ft]
        schedule.append({'event_id':f'E{i+1:03d}','site_id':site,'fault_type':ft,'start_day':start_day,
                         'start_hour':float(rng.uniform(5,20)),'duration_hours':dur,'severity':0.45,'window':'train_faint'})
    # ensure all types in eval, then fill
    eid=len(schedule)+1
    eval_types=fault_types + list(rng.choice(fault_types, size=cfg['faults_total']-len(schedule)-len(fault_types), replace=True))
    rng.shuffle(eval_types)
    occupied={f"site_{i+1:02d}":[] for i in range(cfg['n_sites'])}
    for ft in eval_types:
        for attempt in range(100):
            site=f"site_{rng.integers(1,cfg['n_sites']+1):02d}"
            if ft=='F1': dur=float(rng.uniform(7,22))
            elif ft=='F2': dur=float(rng.uniform(14,42))
            elif ft=='F3': dur=float(rng.uniform(2.5,7.5))
            elif ft=='F4': dur=float(rng.uniform(1.5,5.5))
            elif ft=='F5': dur=float(rng.uniform(0.75,3.0))
            else: dur=float(rng.uniform(2,5))
            start_day=int(rng.integers(11,21))
            start_hour=float(rng.uniform(0,22))
            st=(start_day-1)*24+start_hour; en=st+dur
            if all(en < a-3 or st > b+3 for a,b in occupied[site]):
                occupied[site].append((st,en)); break
        schedule.append({'event_id':f'E{eid:03d}','site_id':site,'fault_type':ft,'start_day':start_day,
                         'start_hour':start_hour,'duration_hours':dur,'severity':float(rng.uniform(0.75,1.15)),'window':'eval'})
        eid+=1
    return schedule

def inject(df,cfg,schedule):
    out=df.copy()
    start0=pd.Timestamp(cfg['start_ts'])
    labels=[]
    for ev in schedule:
        st=start0 + pd.Timedelta(days=ev['start_day']-1, hours=ev['start_hour'])
        en=st + pd.Timedelta(hours=ev['duration_hours'])
        mask=(out.site_id==ev['site_id']) & (out.timestamp>=st) & (out.timestamp<=en)
        idx=out.index[mask]
        if len(idx)==0: continue
        x=np.linspace(0,1,len(idx)); sev=ev['severity']; ft=ev['fault_type']
        if ft=='F1':
            ramp=x*sev
            out.loc[idx,'avg_signal_quality_db']-=5.5*ramp
            out.loc[idx,'call_drop_rate_pct']+=1.4*ramp
            out.loc[idx,'retransmission_rate_pct']+=2.6*ramp
            out.loc[idx,'latency_ms']+=8*ramp
        elif ft=='F2':
            ramp=x*sev
            out.loc[idx,'handover_success_rate_pct']-=3.2*ramp
            out.loc[idx,'call_drop_rate_pct']+=0.35*ramp
        elif ft=='F3':
            osc=np.sin(np.arange(len(idx))*np.pi/2)*sev
            rng=np.random.default_rng(cfg['seed']+len(idx))
            out.loc[idx,'supply_voltage_v']+=0.45*osc + rng.normal(0,0.18,len(idx))
            out.loc[idx,'unit_temperature_c']+=rng.normal(1.8,2.4,len(idx))*sev
            out.loc[idx,'cpu_load_pct']+=rng.normal(6,8,len(idx))*sev
            out.loc[idx,'latency_ms']+=rng.normal(3,6,len(idx))*sev
        elif ft=='F4':
            out.loc[idx,'downlink_traffic_gb']*=0.06*sev
            out.loc[idx,'active_users']*=0.08*sev
            # quality stays apparently fine
            out.loc[idx,'call_setup_success_rate_pct']+=np.minimum(0.5, sev*0.2)
        elif ft=='F5':
            jump=sev
            out.loc[idx,'uplink_interference_dbm']+=9*jump
            out.loc[idx,'avg_signal_quality_db']-=4.5*jump
            out.loc[idx,'call_setup_success_rate_pct']-=2.5*jump
            out.loc[idx,'handover_success_rate_pct']-=2.2*jump
            out.loc[idx,'retransmission_rate_pct']+=3.2*jump
            out.loc[idx,'latency_ms']+=12*jump
        elif ft=='F6':
            surge=(1+0.75*sev)*(0.8+0.2*np.sin(np.linspace(0,np.pi,len(idx))))
            out.loc[idx,'downlink_traffic_gb']*=surge
            out.loc[idx,'active_users']*=surge
            out.loc[idx,'latency_ms']+=12*sev
            out.loc[idx,'cpu_load_pct']+=18*sev
            out.loc[idx,'call_drop_rate_pct']+=0.8*sev
            out.loc[idx,'call_setup_success_rate_pct']-=1.0*sev
        labels.append({'event_id':ev['event_id'],'site_id':ev['site_id'],'fault_type':ft,
                       'start_ts':st.isoformat(),'end_ts':en.isoformat(),
                       'affected_kpis':';'.join(FAULT_KPIS[ft]),'window':ev['window']})
    # realistic bounds
    out['call_setup_success_rate_pct']=out['call_setup_success_rate_pct'].clip(70,99.99)
    out['handover_success_rate_pct']=out['handover_success_rate_pct'].clip(70,99.99)
    out['call_drop_rate_pct']=out['call_drop_rate_pct'].clip(0,25)
    out['retransmission_rate_pct']=out['retransmission_rate_pct'].clip(0,35)
    out['cpu_load_pct']=out['cpu_load_pct'].clip(0,100)
    out['active_users']=out['active_users'].clip(0,None)
    out['downlink_traffic_gb']=out['downlink_traffic_gb'].clip(0,None)
    return out, pd.DataFrame(labels)

def add_missing(df,cfg):
    rng=np.random.default_rng(cfg['seed']+200)
    out=df.copy()
    for site in out.site_id.unique():
        rate=float(rng.uniform(cfg['missing_rate_min'], cfg['missing_rate_max']))
        mask_site=out.site_id==site
        for col in KPIS:
            eligible=out.index[mask_site]
            miss=rng.choice(eligible, size=max(1,int(len(eligible)*rate)), replace=False)
            out.loc[miss,col]=np.nan
    return out

def main(config_path='code/config.yaml', root='.'):
    cfg=load_config(os.path.join(root, config_path) if not os.path.isabs(config_path) else config_path)
    df=create_base(cfg)
    schedule=random_fault_schedule(cfg)
    df, labels=inject(df,cfg,schedule)
    df=add_missing(df,cfg)
    os.makedirs(os.path.join(root,'data'), exist_ok=True)
    os.makedirs(os.path.join(root,'ground_truth'), exist_ok=True)
    df.to_csv(os.path.join(root,'data','telemetry.csv'), index=False)
    labels.to_csv(os.path.join(root,'ground_truth','fault_events.csv'), index=False)
    with open(os.path.join(root,'ground_truth','fault_schedule.json'),'w') as f: json.dump(schedule,f,indent=2)
    print(f'Generated {len(df)} rows and {len(labels)} fault events')

if __name__ == '__main__':
    main(root=os.path.abspath(os.path.join(os.path.dirname(__file__),'..')))
