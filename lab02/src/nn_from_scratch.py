from pathlib import Path
import csv, time
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "lab02" / "results"
OUT.mkdir(parents=True, exist_ok=True)
SEED, BATCH, EPOCHS, EPS = 42, 32, 50, 1e-5
WIDTHS, RATES = [16, 64], [0.001, 0.01, 0.1]


def init_net(w, seed=SEED):
    r = np.random.default_rng(seed)
    # N(0, 0.01): variance 0.01, so standard deviation is sqrt(0.01)=0.1.
    return {"W1": r.normal(0, .1, (64,w)), "b1": np.zeros(w),
            "W2": r.normal(0, .1, (w,w)), "b2": np.zeros(w),
            "W3": r.normal(0, .1, (w,10)), "b3": np.zeros(10)}


def forward(x, p):
    z1 = x @ p["W1"] + p["b1"]; a1 = np.maximum(z1, 0)
    z2 = a1 @ p["W2"] + p["b2"]; a2 = np.maximum(z2, 0)
    z3 = a2 @ p["W3"] + p["b3"]
    e = np.exp(z3 - z3.max(axis=1, keepdims=True))
    probs = e / e.sum(axis=1, keepdims=True)
    return probs, (x, z1, a1, z2, a2, probs)


def loss(probs, y):
    return float(-np.sum(y * np.log(np.clip(probs, 1e-12, 1))) / len(y))


def backward(y, p, c):
    x,z1,a1,z2,a2,probs = c; n = len(y)
    dz3 = (probs-y)/n
    dw3 = a2.T @ dz3; db3 = dz3.sum(axis=0)
    dz2 = (dz3 @ p["W3"].T) * (z2 > 0)
    dw2 = a1.T @ dz2; db2 = dz2.sum(axis=0)
    dz1 = (dz2 @ p["W2"].T) * (z1 > 0)
    dw1 = x.T @ dz1; db1 = dz1.sum(axis=0)
    return {"W1":dw1,"b1":db1,"W2":dw2,"b2":db2,"W3":dw3,"b3":db3}


def grad_check(x, y):
    p = init_net(16, 2026)
    _, c = forward(x, p); g = backward(y, p, c)
    rng = np.random.default_rng(123)
    choices = [(k, ix) for k in ("W1","W2","W3") for ix in np.ndindex(p[k].shape)]
    selected = rng.choice(len(choices), 10, replace=False)
    rows = []
    for num, j in enumerate(selected, 1):
        k, ix = choices[int(j)]; old = p[k][ix]
        p[k][ix] = old + EPS; lp = loss(forward(x,p)[0],y)
        p[k][ix] = old - EPS; lm = loss(forward(x,p)[0],y)
        p[k][ix] = old
        ng = (lp-lm)/(2*EPS); ag = float(g[k][ix])
        err = abs(ag-ng)/max(1e-12,abs(ag)+abs(ng))
        rows.append({"weight":num,"parameter":k,"index":str(ix),
                     "analytical":ag,"numerical":ng,"relative_error":err,
                     "passed":err < 1e-6})
    return rows


def train(x, y, w, lr, seed):
    p = init_net(w, seed); rng = np.random.default_rng(seed+100)
    curve = []
    for epoch in range(EPOCHS):
        order = rng.permutation(len(x))
        for start in range(0,len(x),BATCH):
            ids = order[start:start+BATCH]
            probs,c = forward(x[ids],p); grads = backward(y[ids],p,c)
            for k in p: p[k] -= lr*grads[k]
        curve.append(loss(forward(x,p)[0],y))
        print(f"W={w}, lr={lr:g}, epoch {epoch+1:02d}/{EPOCHS}, loss={curve[-1]:.5f}")
    return p,curve


def write_csv(path, rows):
    with path.open("w",newline="",encoding="utf-8") as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0]))
        writer.writeheader(); writer.writerows(rows)


def main():
    d=load_digits()
    X=d.data.astype(np.float64)/16.0
    y=np.eye(10,dtype=np.float64)[d.target]
    Xtr,Xte,ytr,yte,ytr_oh,yte_oh=train_test_split(
        X,d.target,y,test_size=.30,stratify=d.target,random_state=SEED)
    p=init_net(16); probs,_=forward(Xtr[:32],p)
    assert probs.shape==(32,10) and np.allclose(probs.sum(1),1,atol=1e-6)
    initial=loss(probs,ytr_oh[:32])
    print(f"Initial loss {initial:.6f}; ln(10)={np.log(10):.6f}")

    gc=grad_check(Xtr[:5],ytr_oh[:5]); write_csv(OUT/"gradient_check.csv",gc)
    passed=sum(r["passed"] for r in gc)
    print(f"Gradient check: {passed}/10 passed (need at least 9).")

    results=[]; curves={}
    for w in WIDTHS:
        for lr in RATES:
            p,curve=train(Xtr,ytr_oh,w,lr,SEED+w)
            acc=accuracy_score(yte,np.argmax(forward(Xte,p)[0],axis=1))
            curves[f"W={w}, lr={lr:g}"]=curve
            results.append({"hidden_width":w,"learning_rate":lr,
                            "final_train_loss":curve[-1],"test_accuracy":acc})
            print(f"Test accuracy W={w}, lr={lr:g}: {acc:.4f}")
    write_csv(OUT/"experiment_results.csv",results)

    plt.figure(figsize=(10,6))
    for label,curve in curves.items(): plt.plot(range(1,EPOCHS+1),curve,label=label)
    plt.xlabel("Epoch"); plt.ylabel("Training cross-entropy loss")
    plt.title("Lab 2 loss curves"); plt.grid(alpha=.3); plt.legend()
    plt.tight_layout(); plt.savefig(OUT/"loss_curves.png",dpi=160); plt.close()

    timing=[]
    for w in WIDTHS:
        p=init_net(w); xb=Xtr[:BATCH]; yb=ytr_oh[:BATCH]
        for _ in range(5):
            _,c=forward(xb,p); backward(yb,p,c)
        ft=[]; bt=[]
        for _ in range(100):
            t=time.perf_counter(); _,c=forward(xb,p); ft.append(time.perf_counter()-t)
            t=time.perf_counter(); backward(yb,p,c); bt.append(time.perf_counter()-t)
        timing.append({"hidden_width":w,"batch_size":BATCH,"runs":100,
                       "forward_median_ms":np.median(ft)*1000,
                       "backward_median_ms":np.median(bt)*1000})
    write_csv(OUT/"timing.csv",timing)

    lines=["# Lab 2: A Neural Network from Scratch","",
    "## Dataset and preprocessing",
    "- scikit-learn Digits dataset; 64 inputs scaled by dividing by 16.",
    "- Labels one-hot encoded; stratified 70/30 split; random_state=42.","",
    "## Architecture",
    "- 64 -> W -> ReLU -> W -> ReLU -> 10 -> softmax; W=16 and W=64.",
    "- Weight shapes: (64,W), (W,W), (W,10); biases: (W,), (W,), (10,).",
    "- Weights sampled from N(0,0.01), interpreted as variance 0.01 (standard deviation 0.1); biases initialized to zero.","",
    "## Forward pass and loss",
    f"- Output shape: {probs.shape}; row sums checked within 1e-6.",
    f"- Initial loss: {initial:.6f}; ln(10)={np.log(10):.6f}.","",
    "## Gradient checking",f"- {passed}/10 selected weights passed relative error < 1e-6.",
    "- Full details: results/gradient_check.csv.","",
    "## Training experiments",
    "- Mini-batch SGD; batch size 32; 50 epochs; shuffled each epoch.",
    "- Learning rates 0.001, 0.01, 0.1; hidden widths 16 and 64.",
    "| Width | Learning rate | Final train loss | Test accuracy |",
    "|---:|---:|---:|---:|"]
    for r in results:
        lines.append(f"| {r['hidden_width']} | {r['learning_rate']:g} | {r['final_train_loss']:.6f} | {r['test_accuracy']:.4f} |")
    lines += ["","Loss plot: `results/loss_curves.png`.","",
    "## Timing and analysis",
    "| Width | Batch size | Forward median (ms) | Backward median (ms) |",
    "|---:|---:|---:|---:|"]
    for row in timing:
        lines.append(f"| {row['hidden_width']} | {row['batch_size']} | {row['forward_median_ms']:.6f} | {row['backward_median_ms']:.6f} |")
    lines += ["",
    "- Timings are medians over 100 runs for each hidden width.",
    "- Both passes use matrix multiplications. Forward propagation computes activations and logits. Backpropagation computes weight gradients and propagates error signals through the layers. Compare the measured medians above to identify which step dominated in this run.","",
    "## Implementation",
    "- Network, softmax, cross-entropy, gradients, and training loop implemented with NumPy; no PyTorch or TensorFlow used."]
    (ROOT/"lab02"/"report.md").write_text("\n".join(lines)+"\n",encoding="utf-8")
    print("\nLab 2 finished. Report:",ROOT/"lab02"/"report.md")
    print("Evidence:",OUT)


if __name__=="__main__":
    main()
