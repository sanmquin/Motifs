import json, os, textwrap, random
from pathlib import Path

ROOT=Path('.')
NB_DIR=ROOT/'convolution_transformer_arc_notebooks'
DATA_DIR=ROOT/'convolution_transformer_arc_synthetic_data'
NB_DIR.mkdir(exist_ok=True); DATA_DIR.mkdir(exist_ok=True)

def md(title, body):
    return {"cell_type":"markdown","metadata":{},"source":[f"# {title}\n\n", body.strip()+"\n"]}
def code(title, desc, src):
    return {"cell_type":"code","execution_count":None,"metadata":{},"outputs":[],"source":[f"# {title}\n", f"# {desc}\n\n", textwrap.dedent(src).strip()+"\n"]}

COMMON_IMPORTS=r'''
import importlib.util, json, math, random, os, sys, subprocess
from pathlib import Path
from dataclasses import dataclass

if importlib.util.find_spec('numpy') is None or importlib.util.find_spec('matplotlib') is None:
    subprocess.check_call([sys.executable, '-m', 'pip', 'install', 'numpy==1.26.4', 'matplotlib==3.8.4'])
if importlib.util.find_spec('torch') is None:
    subprocess.check_call([sys.executable, '-m', 'pip', 'install', 'torch==2.2.2'])

import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

IN_COLAB = importlib.util.find_spec('google.colab') is not None
if IN_COLAB:
    from google.colab import drive
    drive.mount('/content/drive')

DRIVE_ROOT = Path('/content/drive/MyDrive/motifs') if IN_COLAB else Path.cwd().parent
LOCAL_EXPORT_ROOT = DRIVE_ROOT / 'convolution_transformer_arc_synthetic_data'
LOCAL_EXPORT_ROOT.mkdir(parents=True, exist_ok=True)
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
PALETTE_SIZE = 10
PAD = 10
random.seed(7); np.random.seed(7); torch.manual_seed(7)
print({'in_colab': IN_COLAB, 'device': DEVICE, 'export_root': str(LOCAL_EXPORT_ROOT)})
'''

MODEL_CODE=r'''
class ConvTransformer(nn.Module):
    def __init__(self, colors=11, d_model=96, heads=4, layers=3, max_hw=30):
        super().__init__()
        self.embed = nn.Embedding(colors, d_model)
        self.local = nn.Sequential(
            nn.Conv2d(d_model, d_model, 3, padding=1, groups=1), nn.GELU(),
            nn.Conv2d(d_model, d_model, 3, padding=1), nn.GELU())
        self.pos_r = nn.Embedding(max_hw, d_model)
        self.pos_c = nn.Embedding(max_hw, d_model)
        enc = nn.TransformerEncoderLayer(d_model=d_model, nhead=heads, dim_feedforward=4*d_model, batch_first=True, dropout=0.0)
        self.transformer = nn.TransformerEncoder(enc, num_layers=layers)
        self.head = nn.Conv2d(d_model, colors-1, 1)
    def forward(self, x):
        b,h,w=x.shape
        z=self.embed(x).permute(0,3,1,2)
        z=self.local(z)
        rr=torch.arange(h,device=x.device); cc=torch.arange(w,device=x.device)
        pos=(self.pos_r(rr)[:,None,:]+self.pos_c(cc)[None,:,:]).permute(2,0,1)[None]
        z=z+pos
        seq=z.flatten(2).transpose(1,2)
        seq=self.transformer(seq)
        z=seq.transpose(1,2).reshape(b,-1,h,w)
        return self.head(z)

class GridPairs(Dataset):
    def __init__(self, pairs): self.pairs=pairs
    def __len__(self): return len(self.pairs)
    def __getitem__(self, i):
        x=np.array(self.pairs[i]['input'], dtype=np.int64)
        y=np.array(self.pairs[i]['output'], dtype=np.int64)
        return torch.tensor(x), torch.tensor(y)

def train_model(splits, epochs=12, batch_size=32, lr=3e-4):
    model=ConvTransformer().to(DEVICE); opt=torch.optim.AdamW(model.parameters(),lr=lr,weight_decay=1e-4)
    loaders={k:DataLoader(GridPairs(v),batch_size=batch_size,shuffle=(k=='train')) for k,v in splits.items()}
    history=[]; epoch_outputs=[]
    for ep in range(1, epochs+1):
        model.train(); losses=[]
        for x,y in loaders['train']:
            x,y=x.to(DEVICE),y.to(DEVICE); logits=model(x); loss=F.cross_entropy(logits,y)
            opt.zero_grad(); loss.backward(); opt.step(); losses.append(loss.item())
        row={'epoch':ep,'train_loss':float(np.mean(losses))}
        model.eval()
        with torch.no_grad():
            for split in ['val','test']:
                correct=total=0; exact=0; count=0
                for x,y in loaders[split]:
                    x,y=x.to(DEVICE),y.to(DEVICE); pred=model(x).argmax(1)
                    correct+=(pred==y).sum().item(); total+=y.numel(); exact+=(pred==y).flatten(1).all(1).sum().item(); count+=y.shape[0]
                row[f'{split}_pixel_acc']=correct/total; row[f'{split}_exact_acc']=exact/count
            sx,sy=next(iter(loaders['test'])); pred=model(sx.to(DEVICE)).argmax(1).cpu().numpy()[0]
            epoch_outputs.append({'epoch':ep,'input':sx.numpy()[0].tolist(),'target':sy.numpy()[0].tolist(),'prediction':pred.tolist()})
        history.append(row); print(row)
    return model, history, epoch_outputs

def plot_history(history):
    epochs=[r['epoch'] for r in history]
    fig,ax=plt.subplots(1,2,figsize=(12,4))
    ax[0].plot(epochs,[r['train_loss'] for r in history],marker='o'); ax[0].set_title('Training Loss by Epoch'); ax[0].set_xlabel('Epoch'); ax[0].set_ylabel('Cross Entropy')
    ax[1].plot(epochs,[r['val_exact_acc'] for r in history],label='Validation exact',marker='o')
    ax[1].plot(epochs,[r['test_exact_acc'] for r in history],label='Test exact',marker='o')
    ax[1].set_title('Hold-out Exact-grid Accuracy'); ax[1].set_xlabel('Epoch'); ax[1].set_ylabel('Accuracy'); ax[1].legend(); plt.show()

def show_triplet(sample):
    fig,ax=plt.subplots(1,3,figsize=(9,3))
    for a,k in zip(ax,['input','target','prediction']):
        a.imshow(sample[k],vmin=0,vmax=9,cmap='tab10'); a.set_title(k); a.set_xticks([]); a.set_yticks([])
    plt.show()
'''

def make_0(seed=0, n=320):
    rng=random.Random(seed); pairs=[]
    for _ in range(n):
        g=[[0]*10 for _ in range(10)]; out=[[0]*10 for _ in range(10)]; colors=rng.sample(range(1,10),3)
        sizes=[1,2,3]
        locs=[]
        for color,sz in zip(colors,sizes):
            placed=False
            while not placed:
                r=rng.randrange(0,10-sz); c=rng.randrange(0,10-sz)
                if all(g[rr][cc]==0 for rr in range(max(0,r-1),min(10,r+sz+1)) for cc in range(max(0,c-1),min(10,c+sz+1))):
                    for rr in range(r,r+sz):
                        for cc in range(c,c+sz):
                            g[rr][cc]=color; out[rr][cc]=colors[sz-1]
                    placed=True
        pairs.append({'input':g,'output':out})
    return pairs

def make_1(seed=0,n=320):
    rng=random.Random(seed); pairs=[]
    for _ in range(n):
        g=[[0]*10 for _ in range(10)]; out=[[0]*10 for _ in range(10)]; color=rng.randrange(1,10)
        r=rng.randrange(2,8); c=rng.randrange(2,8); g[r][c]=color
        for i in range(10): out[r][i]=color; out[i][c]=color
        blocker=rng.randrange(1,10)
        if blocker==color: blocker=(blocker%9)+1
        for _ in range(rng.randrange(1,4)):
            br=rng.randrange(10); bc=rng.randrange(10); g[br][bc]=blocker; out[br][bc]=blocker
        pairs.append({'input':g,'output':out})
    return pairs

PUZZLES={
 '0a2355a6':('Component-size color ordering can be learned from local shapes while preserving ARC colors.', make_0),
 '1acc24af':('Cross-completion from a colored seed can be learned with convolutional attention over rows and columns.', make_1),
}
for pid,(hyp,fn) in PUZZLES.items():
    pdir=DATA_DIR/pid; pdir.mkdir(parents=True,exist_ok=True)
    pairs=fn(7,360); splits={'train':pairs[:280],'val':pairs[280:320],'test':pairs[320:]}
    for k,v in splits.items(): (pdir/f'{k}.json').write_text(json.dumps(v))
    cells=[md(f'Hypothesis: {hyp}', f'This notebook solves ARC puzzle `{pid}` using only input-output pairs generated by a task-specific synthetic data engine. The experiment tests whether a compact convolution transformer improves visual reasoning by combining local convolutional shape priors with global self-attention. Colors remain the original integer ARC labels throughout training; no palette remapping is learned.'),
           md('Methodology: Synthetic input-output pairs, hold-outs, and convolution transformers', 'We first generate synthetic grids that instantiate the same transformation family as the target ARC puzzle, then split them into training, validation, and test hold-outs. The neural solver is trained only on input grids paired with output grids. After every epoch, the notebook logs scalar metrics and a representative prediction so qualitative errors can be inspected alongside charts.'),
           md('Hypotheses: Neural visual reasoning improves with convolutional attention', 'Null hypothesis: the convolution transformer does not learn the puzzle transformation beyond chance pixel accuracy on validation and test hold-outs. Alternative hypothesis: the model learns the transformation with high exact-grid accuracy on held-out synthetic examples while preserving the constant ARC color vocabulary.'),
           code('Environment Setup','Mount Google Drive in Colab, import libraries, define export paths, and keep dependency installation compatible with Colab.', COMMON_IMPORTS),
           code('Synthetic Data Generator','Create puzzle-specific input-output pairs and export train, validation, and test JSON files to a new directory.', f"""
PUZZLE_ID='{pid}'
{fn.__name__ == 'make_0' and 'generate_pairs = '+make_0.__name__ or 'generate_pairs = '+make_1.__name__}
# generator definition is embedded below for reproducibility
"""),
          ]
    # replace generator cell with actual source
    gen_src=(r'''
def generate_pairs(seed=7, n=360):
    rng=random.Random(seed); pairs=[]
    for _ in range(n):
        g=np.zeros((10,10), dtype=int); out=np.zeros_like(g); colors=rng.sample(range(1,10),3)
        for color,sz in zip(colors,[1,2,3]):
            while True:
                r=rng.randrange(0,10-sz); c=rng.randrange(0,10-sz)
                if g[max(0,r-1):r+sz+1,max(0,c-1):c+sz+1].sum()==0:
                    g[r:r+sz,c:c+sz]=color; out[r:r+sz,c:c+sz]=colors[sz-1]; break
        pairs.append({'input':g.tolist(),'output':out.tolist()})
    return pairs
''' if pid=='0a2355a6' else r'''
def generate_pairs(seed=7, n=360):
    rng=random.Random(seed); pairs=[]
    for _ in range(n):
        g=[[0]*10 for _ in range(10)]; out=[[0]*10 for _ in range(10)]; color=rng.randrange(1,10)
        r=rng.randrange(2,8); c=rng.randrange(2,8); g[r][c]=color
        for i in range(10): out[r][i]=color; out[i][c]=color
        blocker=rng.randrange(1,10)
        if blocker==color: blocker=(blocker%9)+1
        for _ in range(rng.randrange(1,4)):
            br=rng.randrange(10); bc=rng.randrange(10); g[br][bc]=blocker; out[br][bc]=blocker
        pairs.append({'input':g,'output':out})
    return pairs
''')+r'''
pairs=generate_pairs(); splits={'train':pairs[:280],'val':pairs[280:320],'test':pairs[320:]}
EXPORT_DIR=LOCAL_EXPORT_ROOT/PUZZLE_ID; EXPORT_DIR.mkdir(parents=True, exist_ok=True)
for name,rows in splits.items():
    (EXPORT_DIR/f'{name}.json').write_text(json.dumps(rows))
print({k:len(v) for k,v in splits.items()}, EXPORT_DIR)
'''
    cells[-1]=code('Synthetic Data Generator','Generate synthetic examples for this puzzle, keep color IDs constant, and export each split.', gen_src)
    cells += [code('Convolution Transformer Architecture','Define a hybrid model with convolutional local features, row/column positional embeddings, global transformer layers, and per-cell color logits.', MODEL_CODE),
              code('Training Loop and Epoch Logging','Train only on input-output pairs, evaluate hold-outs, and store representative outputs after each epoch.', """model, history, epoch_outputs = train_model(splits, epochs=12, batch_size=32)
(Path(EXPORT_DIR)/'epoch_outputs.json').write_text(json.dumps(epoch_outputs))
(Path(EXPORT_DIR)/'metrics.json').write_text(json.dumps(history))"""),
              md('Results: Charts of loss, validation accuracy, and test accuracy','The following charts present quantitative training results. Decreasing cross-entropy indicates better per-pixel color prediction; exact-grid accuracy is the stricter ARC-style metric because every output cell must be correct.'),
              code('Results Visualization','Plot training loss and exact-grid hold-out accuracy.', 'plot_history(history)'),
              md('Interpretation: How to read epoch outputs','The qualitative panels below show whether the network learned the intended transformation rather than merely improving average pixel accuracy. Inspect the prediction after the final epoch for shape placement, row-column propagation, and color preservation errors.'),
              code('Qualitative Epoch Output','Display the final logged input, target, and prediction.', 'show_triplet(epoch_outputs[-1])')]
    nb={"cells":cells,"metadata":{"kernelspec":{"display_name":"Python 3","language":"python","name":"python3"},"language_info":{"name":"python","version":"3.10"}},"nbformat":4,"nbformat_minor":5}
    (NB_DIR/f'{pid}_convolution_transformer_solver.ipynb').write_text(json.dumps(nb, indent=2))
print('generated')
