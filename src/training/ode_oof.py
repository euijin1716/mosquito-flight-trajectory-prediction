"""
ode_oof.py (구 train_phaseODE.py) — Neural ODE 5-fold OOF 학습
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
각 11스텝 궤적에서 내부 지점 e의 "X[0..e] → X[e+2](+80ms)" 전이를 추가 학습.
real(e=10, 정답 라벨) + interior(e∈{5,6,7,8}, 관측된 X[e+2]) = 5 examples/traj.
가변길이 윈도우 → 좌측 zero-pad + masked-mean. 마지막 스텝(예측 기준)은 항상 유효.
base=cv_1step(window), target=X[e+2]-base (rotated). norm은 norm_stats 사용.

평가: REAL 예시(전체 11윈도우→라벨)의 OOF R-Hit.
실행: python -m src.training.ode_oof
"""
import sys, time, logging, argparse, subprocess
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
import numpy as np, pandas as pd
import os
import torch, torch.nn as nn, torch.nn.functional as F
HID=int(os.environ.get('HID',128)); NL=int(os.environ.get('NL',3)); DR=float(os.environ.get('DR','0.15'))
HEADW=int(os.environ.get('HEADW',256)); TAG=os.environ.get('TAG','g')
HW=float(os.environ.get('HW','0.5')); GW=float(os.environ.get('GW','0.5'))
POOL=os.environ.get('POOL','attn')  # attn(기본) | mean
ENC=os.environ.get('ENC','gru')   # gru(기본) | tcn
from sklearn.model_selection import KFold

ROOT = Path(__file__).resolve().parents[2]; sys.path.insert(0, str(ROOT))
from src.features.base import yaw_rotation_matrix, extract_seq_features, extract_scalar_features, load_sample
from src.features.norm_stats import load_or_build as _load_norm_stats
EXP = ROOT/'experiments'; MODEL = ROOT/'models'
DT = 0.04; K_FOLDS=5; EPOCHS=120; LR=2e-4; WD=1e-4; PATIENCE=30; BATCH=256; SEED=42
SIGMA=0.02; RHIT_TAU=float(os.environ.get('RT','0.003')); RHIT_W=float(os.environ.get('RW','0.5')); FLIP_PROB=0.5; NOISE_STD=0.02
Y_FLIP=[1,4,7,10]; INTERIOR_E=[5,6,7,8]
SPEED_BINS=[0,0.3,0.6,0.9,1.2,float('inf')]; SPEED_LAB=['0~0.3','0.3~0.6','0.6~0.9','0.9~1.2','1.2+']
CACHE = EXP/'train_cache.npz'

def get_device():
    return (torch.device('cuda') if torch.cuda.is_available()
            else torch.device('mps') if torch.backends.mps.is_available() else torch.device('cpu'))
def get_logger(name, path):
    lg=logging.getLogger(name); lg.setLevel(logging.INFO)
    if not lg.handlers:
        f=logging.Formatter('%(asctime)s | %(message)s','%H:%M:%S')
        for h in [logging.StreamHandler(sys.stdout), logging.FileHandler(path,'w','utf-8')]:
            h.setFormatter(f); lg.addHandler(h)
    return lg
def r_hit(p,t,thr=0.01): return float(np.mean(np.linalg.norm(p-t,axis=1)<=thr))
def speed_bin_rhit(p,t,s):
    o={}
    for lab,lo,hi in zip(SPEED_LAB,SPEED_BINS[:-1],SPEED_BINS[1:]):
        m=(s>=lo)&(s<hi); o[lab]=r_hit(p[m],t[m]) if m.sum() else float('nan')
    return o

def window_features(W):
    """길이 L≥4 윈도우 → seq(L,13), scalar(22), rot, base, last_pos. L=11이면 features_clean과 동일."""
    W=W.astype(np.float64); vel=np.gradient(W,DT,axis=0); rot=yaw_rotation_matrix(vel[-1])
    seq=extract_seq_features(W,vel,rot); b14=extract_scalar_features(W,vel)
    sp=np.linalg.norm(vel,axis=1); steps=np.linalg.norm(np.diff(W,axis=0),axis=1); L=len(W)
    path=float(steps.sum()); net=float(np.linalg.norm(W[-1]-W[0])); straight=net/(path+1e-8)
    t=np.arange(float(L)); noise=float(np.mean([(W[:,d]-np.polyval(np.polyfit(t,W[:,d],2),t)).std() for d in range(3)]))
    k=min(4,L); acc_trend=float(np.polyfit(np.arange(float(k)),sp[-k:],1)[0])
    sc=np.concatenate([b14,[float(sp.max()),float(sp.std()),float(sp[-3:].mean()),float(sp[-5:].mean()),
                            path,straight,noise,acc_trend]]).astype(np.float32)
    base=(W[-1]+2.0*(W[-1]-W[-2])).astype(np.float32)
    return seq.astype(np.float32), sc, rot.astype(np.float32), base, W[-1].astype(np.float32)

def build_cache():
    st=_load_norm_stats(EXP/'norm_stats.npz'); sm,ss,cm,cs=st['seq_mean'],st['seq_std'],st['scalar_mean'],st['scalar_std']
    paths=sorted((ROOT/'Data'/'train').glob('*.csv'))
    labs=pd.read_csv(ROOT/'Data'/'train_labels.csv').sort_values('id').reset_index(drop=True)[['x','y','z']].to_numpy(np.float32)
    SEQ=[]; SCAL=[]; MASK=[]; TGT=[]; ROT=[]; SPD=[]; TRAJ=[]; REAL=[]
    for i,p in enumerate(paths):
        X=load_sample(p).astype(np.float64)
        ex=[(10, labs[i])]                       # real
        ex+=[(e, X[e+2]) for e in INTERIOR_E]    # interior
        for e,tgt in ex:
            W=X[:e+1]; L=len(W)
            seq,sc,rot,base,lp=window_features(W)
            seq_n=((seq-sm)/ss).astype(np.float32); sc_n=((sc-cm)/cs).astype(np.float32)
            pad=11-L
            seq11=np.zeros((11,13),np.float32); seq11[pad:]=seq_n
            mask=np.zeros(11,np.float32); mask[pad:]=1.0
            tgt_rot=(rot@(tgt-base)).astype(np.float32)
            SEQ.append(seq11); SCAL.append(sc_n); MASK.append(mask); TGT.append(tgt_rot); ROT.append(rot)
            SPD.append(float(np.linalg.norm(np.gradient(W,DT,axis=0)[-1]))); TRAJ.append(i); REAL.append(int(e==10))
    d=dict(seq=np.stack(SEQ),scal=np.stack(SCAL),mask=np.stack(MASK),tgt=np.stack(TGT),rot=np.stack(ROT),
           base=None,spd=np.array(SPD,np.float32),traj=np.array(TRAJ),real=np.array(REAL),labels=labs)
    # base/true_pos for real eval: recompute base for real examples on the fly in eval via rot+tgt+label
    np.savez_compressed(CACHE, seq=d['seq'],scal=d['scal'],mask=d['mask'],tgt=d['tgt'],rot=d['rot'],
                        spd=d['spd'],traj=d['traj'],real=d['real'],labels=labs)
    print(f'train_cache: {len(SEQ):,} examples (real {sum(REAL):,})')

def combined_loss(pred,true):
    d=0.01; hub=F.huber_loss(pred,true,delta=d)/(0.5*d*d)
    d2=(pred-true).pow(2).sum(-1); soft=(1-torch.exp(-d2/(2*SIGMA**2))).mean()
    dd=torch.sqrt(d2+1e-12); sr=-torch.sigmoid((0.01-dd)/RHIT_TAU).mean()
    return HW*hub+GW*soft+RHIT_W*sr

class MaskedBiGRU(nn.Module):
    """Neural ODE 잔차 모델: GRU 인코더→latent, 잔차를 댐핑가속도장 RK4 적분으로 누적.
    rpos,rvel=0에서 시작 → a_neural=0이면 잔차=0(=cv예측). 동역학 구조가 GRU와 탈상관."""
    def __init__(self,seq_dim=13,scal_dim=22,h=128,nl=2,dr=0.15,latent=96,nsteps=int(os.environ.get('NSTEPS','4'))):
        super().__init__()
        self.proj=nn.Sequential(nn.Linear(seq_dim,h),nn.LayerNorm(h))
        self.gru=nn.GRU(h,h,nl,batch_first=True,bidirectional=True,dropout=dr if nl>1 else 0)
        self.to_latent=nn.Sequential(nn.Linear(h*4+scal_dim,latent),nn.LayerNorm(latent),nn.GELU())
        self.accel=nn.Sequential(nn.Linear(3+3+latent,128),nn.LayerNorm(128),nn.GELU(),nn.Dropout(dr),
                                 nn.Linear(128,64),nn.GELU(),nn.Linear(64,3))
        self.damping=nn.Parameter(torch.tensor([1.0,1.0,1.0]))
        self.bias=nn.Parameter(torch.zeros(3))
        self.nsteps=nsteps; self.dt=0.08/nsteps
        self._areg=torch.tensor(0.0)
    def _deriv(self,rpos,rvel,lat):
        a=self.accel(torch.cat([rpos,rvel,lat],-1))
        return rvel, -self.damping*rvel+a, a
    def forward(self,seq,scal,mask):
        x=self.proj(seq); out,_=self.gru(x)
        m=mask.unsqueeze(-1); mean=(out*m).sum(1)/m.sum(1).clamp(min=1)
        lat=self.to_latent(torch.cat([out[:,-1,:],mean,scal],-1))
        rpos=torch.zeros(seq.size(0),3,device=seq.device); rvel=torch.zeros_like(rpos)
        areg=0.0
        for _ in range(self.nsteps):
            dt=self.dt
            dp1,dv1,a1=self._deriv(rpos,rvel,lat)
            dp2,dv2,a2=self._deriv(rpos+0.5*dt*dp1,rvel+0.5*dt*dv1,lat)
            dp3,dv3,a3=self._deriv(rpos+0.5*dt*dp2,rvel+0.5*dt*dv2,lat)
            dp4,dv4,a4=self._deriv(rpos+dt*dp3,rvel+dt*dv3,lat)
            rpos=rpos+(dt/6)*(dp1+2*dp2+2*dp3+dp4)
            rvel=rvel+(dt/6)*(dv1+2*dv2+2*dv3+dv4)
            areg=areg+sum(a.pow(2).sum(-1).mean() for a in (a1,a2,a3,a4))/4
        self._areg=areg/self.nsteps
        return rpos+self.bias

def run_fold(fold):
    dev=get_device(); log=get_logger(f'ode_{fold}',EXP/f'log_ode_{fold}.txt')
    d=np.load(CACHE); traj=d['traj']; real=d['real']
    splits=list(KFold(K_FOLDS,shuffle=True,random_state=SEED).split(np.arange(len(d['labels']))))
    tr_traj,va_traj=splits[fold]; tr_set=set(tr_traj.tolist()); va_set=set(va_traj.tolist())
    tr_mask=np.array([t in tr_set for t in traj]); va_mask=(np.array([t in va_set for t in traj]))&(real==1)
    seq=torch.tensor(d['seq']); scal=torch.tensor(d['scal']); msk=torch.tensor(d['mask']); tgt=torch.tensor(d['tgt'])
    tr_idx=np.where(tr_mask)[0]; va_idx=np.where(va_mask)[0]
    rot=d['rot']; spd=d['spd']; labels=d['labels']
    log.info(f'[{TAG}] HID={HID} NL={NL} DR={DR} HEADW={HEADW} RHIT_W={RHIT_W} RHIT_TAU={RHIT_TAU}'); log.info(f'FOLD {fold+1} | train ex={len(tr_idx):,} (real+interior) val real={len(va_idx):,}')
    torch.manual_seed(SEED+fold); model=MaskedBiGRU().to(dev)
    opt=torch.optim.AdamW(model.parameters(),lr=LR,weight_decay=WD)
    sch=torch.optim.lr_scheduler.ReduceLROnPlateau(opt,mode='max',factor=0.5,patience=8,min_lr=1e-6)
    flip=torch.tensor(Y_FLIP,device=dev)
    best=0.0; bep=0; pat=0; path=MODEL/f'ode_fold_{fold}.pt'; t0=time.time()
    EMA_DECAY=0.9; EMA_START=8
    ema={k:v.detach().clone() for k,v in model.state_dict().items()}
    def eval_state(state):
        raw={k:v.detach().clone() for k,v in model.state_dict().items()}
        model.load_state_dict(state); model.eval(); P=[]
        with torch.no_grad():
            for i in range(0,len(va_idx),BATCH):
                b=va_idx[i:i+BATCH]
                pr=model(seq[b].to(dev),scal[b].to(dev),msk[b].to(dev)).cpu().numpy()
                rb=rot[b]; lab=labels[traj[b]]; tgb=d['tgt'][b]
                base=lab-np.einsum('bij,bj->bi',rb.transpose(0,2,1),tgb)
                P.append(base+np.einsum('bij,bj->bi',rb.transpose(0,2,1),pr))
        model.load_state_dict(raw)
        return r_hit(np.concatenate(P), labels[traj[va_idx]])
    # 평가: real 예시만. base=label-rot^T@tgt → pred=base+rot^T@resid. raw·EMA 중 우수 저장.
    for ep in range(1,EPOCHS+1):
        model.train(); np.random.shuffle(tr_idx); tot=0.0
        for i in range(0,len(tr_idx),BATCH):
            b=tr_idx[i:i+BATCH]
            s=seq[b].to(dev); c=scal[b].to(dev); mk=msk[b].to(dev); tg=tgt[b].to(dev)
            if torch.rand(1).item()<FLIP_PROB:
                s=s.clone(); s[:,:,flip]*=-1; tg=tg.clone(); tg[:,1]*=-1
            s=s+torch.randn_like(s)*NOISE_STD*mk.unsqueeze(-1)
            opt.zero_grad(); out=model(s,c,mk); loss=combined_loss(out,tg)+1e-4*model._areg; loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(),0.5); opt.step(); tot+=loss.item()*len(b)
        # EMA 업데이트
        with torch.no_grad():
            for k,v in model.state_dict().items():
                if v.dtype.is_floating_point: ema[k].mul_(EMA_DECAY).add_(v,alpha=1-EMA_DECAY)
                else: ema[k]=v.detach().clone()
        vr=eval_state(model.state_dict())
        vr_e=eval_state(ema) if ep>=EMA_START else 0.0
        use_e=vr_e>vr; vbest=max(vr,vr_e)
        sch.step(vbest); lr=opt.param_groups[0]['lr']
        if vbest>best:
            best,bep,pat=vbest,ep,0
            torch.save({'fold':fold,'epoch':ep,'model_state':(ema if use_e else model.state_dict()),
                        'val_rhit':vbest,'ema':bool(use_e)},path); fl=' ★'+('E' if use_e else '')
        else: pat+=1; fl=''
        log.info(f'ep {ep:3d} | train={tot/len(tr_idx):.4f} raw={vr:.4f} ema={vr_e:.4f}{fl}')
        if pat>=PATIENCE: log.info(f'Early stop (best {bep})'); break
    log.info(f'완료 best R-Hit={best:.4f} ({(time.time()-t0)/60:.1f}min)')

def merge():
    dev=get_device(); d=np.load(CACHE); traj=d['traj']; real=d['real']; labels=d['labels']
    splits=list(KFold(K_FOLDS,shuffle=True,random_state=SEED).split(np.arange(len(labels))))
    seq=torch.tensor(d['seq']); scal=torch.tensor(d['scal']); msk=torch.tensor(d['mask']); rot=d['rot']
    oof=np.zeros((len(labels),3),np.float32); frh=[]
    for fold,(_,va_traj) in enumerate(splits):
        va_set=set(va_traj.tolist()); va_idx=np.where((np.array([t in va_set for t in traj]))&(real==1))[0]
        ck=torch.load(MODEL/f'ode_fold_{fold}.pt',map_location=dev,weights_only=False); frh.append(ck['val_rhit'])
        model=MaskedBiGRU().to(dev); model.load_state_dict(ck['model_state']); model.eval()
        with torch.no_grad():
            for i in range(0,len(va_idx),BATCH):
                b=va_idx[i:i+BATCH]
                pr=model(seq[b].to(dev),scal[b].to(dev),msk[b].to(dev)).cpu().numpy()
                rb=rot[b]; tgb=d['tgt'][b]; lab=labels[traj[b]]
                base=lab-np.einsum('bij,bj->bi',rb.transpose(0,2,1),tgb)
                oof[traj[b]]=base+np.einsum('bij,bj->bi',rb.transpose(0,2,1),pr)
    orh=r_hit(oof,labels)
    print('='*60); print(f'Fold R-Hit: {[f"{r:.4f}" for r in frh]}')
    print(f'ODE OOF R-Hit = {orh:.4f}')
    for lab,rh in speed_bin_rhit(oof,labels,_spd_real(d)).items():
        print(f'  {lab:>8} : {rh:.4f}')
    np.savez(EXP/'ode_oof.npz',preds=oof,true=labels,speeds=_spd_real(d))
    print(f'저장: ode_oof.npz')

def _spd_real(d):
    # 각 traj의 real 예시 speed
    s=np.zeros(len(d['labels']),np.float32); s[d['traj'][d['real']==1]]=d['spd'][d['real']==1]; return s

def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--fold',type=int,default=None); a=ap.parse_args()
    if a.fold is not None: run_fold(a.fold); return
    if not CACHE.exists(): build_cache()
    print(f'Device {get_device()} | 내부전이 사전학습 (e={INTERIOR_E}+real)')
    t0=time.time()
    def sub(f): return f, subprocess.run([sys.executable,str(Path(__file__).resolve()),'--fold',str(f)]).returncode
    with ThreadPoolExecutor(max_workers=K_FOLDS) as ex:
        for fut in as_completed({ex.submit(sub,k):k for k in range(K_FOLDS)}):
            f,c=fut.result(); print(f'  fold {f} {"완료" if c==0 else "오류"} ({(time.time()-t0)/60:.1f}min)')
    merge(); print(f'전체 {(time.time()-t0)/60:.1f}분')

if __name__=='__main__': main()
