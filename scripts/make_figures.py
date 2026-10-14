"""Regenerate the presentation from committed aggregate results; no model loads."""
import json
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
from matplotlib.ticker import FuncFormatter, MaxNLocator
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
DATA = json.loads((ROOT / "evidence/results.json").read_text(encoding="utf-8"))
OUT = ROOT / "assets"
OUT.mkdir(exist_ok=True)
INK, MUTED, BLUE, TEAL, AMBER = "#172B46", "#617189", "#3865D9", "#118777", "#BC6B35"
BG, GRID = "#F7F9FC", "#DDE4EF"
plt.rcParams.update({"font.family":"DejaVu Sans", "font.size":11,
    "figure.facecolor":BG, "axes.facecolor":BG, "text.color":INK,
    "axes.labelcolor":INK, "xtick.color":MUTED, "ytick.color":MUTED,
    "axes.edgecolor":GRID, "axes.spines.top":False, "axes.spines.right":False,
    "svg.fonttype":"none", "svg.hashsalt":"nemesis-showcase-v1"})


def header(fig, chapter, title, subtitle):
    fig.text(.07,.95,chapter.upper(),color=TEAL,size=10,weight="bold",va="top")
    fig.text(.07,.897,title,size=23,weight="bold",va="top")
    fig.text(.07,.828,subtitle,size=11,color=MUTED,va="top")


def footer(fig, text):
    fig.text(.07,.035,text,size=9,color=MUTED,va="bottom")


def save(fig, name):
    for ext in ("svg","png"):
        kwargs = {"metadata":{"Date":None}} if ext == "svg" else {}
        fig.savefig(OUT/f"{name}.{ext}",dpi=170,**kwargs)
        if ext == "svg":
            path=OUT/f"{name}.{ext}"
            path.write_text("\n".join(line.rstrip() for line in path.read_text(encoding="utf-8").splitlines())+"\n",encoding="utf-8",newline="\n")
    plt.close(fig)


def hero():
    fig=plt.figure(figsize=(12,4.8),facecolor=INK)
    fig.text(.065,.85,"NEMESIS",color="white",size=44,weight="bold")
    fig.text(.068,.72,"Learn the player. Test the decision.",color="#D1DDF0",size=20)
    fig.text(.068,.61,"PyTorch  /  CUDA  /  Custom adapters  /  Q-V LoRA",color="#80D9CC",size=12)
    cards=[("8 / 8","players improve","future move prediction"),
           ("17,936","adapter parameters","frozen-backbone training"),
           ("16,384","LoRA parameters","CUDA engineering verified")]
    for x,(value,label,sub) in zip((.065,.37,.675),cards):
        fig.add_artist(FancyBboxPatch((x,.12),.265,.34,boxstyle="round,pad=0.012,rounding_size=0.02",
            transform=fig.transFigure,facecolor="#223C5D",edgecolor="#36516F"))
        fig.text(x+.02,.34,value,color="white",size=28,weight="bold")
        fig.text(x+.02,.245,label,color="#D1DDF0",size=12)
        fig.text(x+.02,.175,sub,color="#9DB5D0",size=10)
    save(fig,"hero")


def prediction():
    fig,ax=plt.subplots(figsize=(12,6.6));fig.subplots_adjust(left=.13,right=.88,bottom=.17,top=.72)
    header(fig,"01 / Prediction","Prediction improves for all eight players",
           "Lower prediction loss is better. Bars show group NLL minus personal NLL.")
    people=DATA["prediction"]["players"];values=[p["nll_gain"] for p in people]
    ax.barh(np.arange(8),values,color=BLUE,height=.57)
    ax.set_yticks(np.arange(8),[p["player"] for p in people]);ax.invert_yaxis()
    ax.set_xlim(0,.15);ax.set_xlabel("Reduction in negative log-likelihood (NLL)",labelpad=10)
    ax.grid(axis="x",color=GRID,alpha=.7);ax.set_axisbelow(True)
    mean=DATA["prediction"]["mean_nll_gain"]
    ax.axvline(mean,color=TEAL,ls="--",lw=1.8,label=f"Mean +{mean:.4f}")
    for i,v in enumerate(values):ax.text(v+.002,i,f"+{v:.4f}",va="center",size=11,weight="bold")
    ax.legend(frameon=False,loc="lower right")
    footer(fig,"8 development players · 7,409 games · 122,593 decisions · game-then-player weighting · future windows H1 + H2")
    save(fig,"prediction_gain")


def identity():
    fig,ax=plt.subplots(figsize=(12,8.7));fig.subplots_adjust(left=.17,right=.84,bottom=.18,top=.75)
    header(fig,"02 / Identity","Each player's own model ranks first",
           "Prediction gain over the shared group model. Diagonal cells match model identity to player.")
    matrix=np.array(DATA["prediction"]["identity_gain_matrix"])
    bound=max(abs(matrix.min()),abs(matrix.max()))
    im=ax.imshow(matrix,cmap="RdBu",vmin=-bound,vmax=bound)
    labels=[p["player"] for p in DATA["prediction"]["players"]]
    ax.set_xticks(range(8),labels);ax.set_yticks(range(8),labels)
    ax.set_xlabel("Player whose future moves are predicted",labelpad=12)
    ax.set_ylabel("Personal model used",labelpad=12)
    for i in range(8):
        for j in range(8):
            ax.text(j,i,f"{matrix[i,j]:+.3f}",ha="center",va="center",fontsize=10,
                color="white" if abs(matrix[i,j])>bound*.53 else INK,weight="bold" if i==j else "normal")
    cbar=fig.colorbar(im,ax=ax,fraction=.045,pad=.04);cbar.set_label("NLL reduction vs. group")
    footer(fig,"Three-seed personal ensembles · all 64 identity pairs · matched future data · positive values favor the personal model")
    save(fig,"identity_prediction")


def training():
    fig,axes=plt.subplots(2,4,figsize=(13,8));fig.subplots_adjust(left=.08,right=.97,bottom=.23,top=.72,hspace=.60,wspace=.35)
    header(fig,"03 / Training","Follow learning through calibration checkpoints",
           "All 24 personal runs. Each curve shows NLL change from its first calibration checkpoint.")
    for ax,p in zip(axes.flat,DATA["prediction"]["players"]):
        runs=[r for r in DATA["calibration_histories"] if r["model_id"].split("_")[1]==p["player"]]
        for run,color in zip(runs,(BLUE,TEAL,AMBER)):
            epochs=run["epochs"];x=[e["epoch"] for e in epochs];raw=np.array([e["calibration_nll"] for e in epochs]);y=raw-raw[0]
            best=int(np.argmin(raw));ax.plot(x,y,"o-",ms=4,lw=1.6,color=color,label=run["model_id"].split("_")[-1])
            ax.scatter(x[best],y[best],s=80,facecolors=BG,edgecolors=color,linewidths=2,zorder=4)
        ax.set_title(p["player"],weight="bold",loc="left",fontsize=12)
        ax.axhline(0,color=GRID);ax.grid(axis="y",color=GRID,alpha=.6)
        ax.xaxis.set_major_locator(MaxNLocator(integer=True));ax.set_xlabel("Epoch")
    axes[0,0].set_ylabel("Change in calibration NLL")
    axes[1,0].set_ylabel("Change in calibration NLL")
    handles,labels=axes[0,0].get_legend_handles_labels()
    fig.legend(handles,["Seed "+x for x in labels],loc="lower center",bbox_to_anchor=(.5,.075),ncol=3,frameon=False)
    footer(fig,"An epoch is a pass through the sampled training data. Hollow markers: selected checkpoints. Different panels use local y scales.")
    save(fig,"training_curves")


def lora():
    q=DATA["lora_engineering"];o=q["overfit"]
    fig,ax=plt.subplots(figsize=(12,6.5));fig.subplots_adjust(left=.1,right=.62,bottom=.17,top=.72)
    header(fig,"04 / LoRA engineering","A low-rank update that learns and resumes exactly",
           "Real Maia-3 backbone · 32 development samples · CUDA FP32 · last two blocks, Q/V only")
    vals=o["losses"]
    ax.plot(range(1,len(vals)+1),vals,"o-",color=BLUE,lw=2.5,ms=5,label="Recorded pre-update losses")
    ax.scatter([len(vals)+1],[o["final_loss"]],marker="D",color=TEAL,s=65,label="Final evaluation")
    ax.set_xlabel("Recorded training step / final evaluation");ax.set_ylabel("Training objective")
    ax.grid(axis="y",color=GRID);ax.legend(frameon=False,fontsize=9,loc="upper right")
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    notes=[("16,384","trainable parameters"),("0.0","zero-init probability error"),("0.0","resume parameter error"),("UNCHANGED","frozen base + K weights")]
    for y,(value,label) in zip((.68,.53,.38,.23),notes):
        fig.text(.70,y,value,size=22,weight="bold",color=TEAL)
        fig.text(.70,y-.045,label,size=10,color=MUTED)
    footer(fig,"Engineering fixture: verifies gradients, freezing and optimizer/RNG recovery. Personalization-effect comparison was not run.")
    save(fig,"lora_engineering")


def decisions():
    fig,axes=plt.subplots(1,2,figsize=(13,6.6));fig.subplots_adjust(left=.12,right=.96,bottom=.25,top=.68,wspace=.34)
    header(fig,"05 / Decisions","Small positive point estimates, evaluated two ways",
           "Personalized selection minus the shared group policy. Dots are means; lines are 95% descriptive intervals.")
    for ax,key,title in zip(axes,("ci95_fixed_players","ci95_player_and_game"),
                            ("Same eight players","Resample players + games")):
        for y,family,color in ((1,"E3",BLUE),(0,"E2",TEAL)):
            row=DATA["decision"][family];low,high=row[key];mean=row["gain"]
            ax.errorbar(mean,y,xerr=np.array([[mean-low],[high-mean]]),fmt="o",color=color,capsize=6,ms=8,lw=2.5)
            ax.annotate(f"{mean:+.6f}",(mean,y),xytext=(0,20),textcoords="offset points",ha="center",weight="bold",color=color)
        ax.set_title(title,loc="left",weight="bold",fontsize=13,pad=22)
        ax.set_yticks([1,0],["Maia-3\nevaluator","Maia-2\nevaluator"])
        ax.set_ylim(-.6,1.6);ax.set_xlim(-.0038,.0040)
        ax.axvline(0,color=MUTED,lw=1.2,ls="--");ax.grid(axis="x",color=GRID)
        ax.set_xlabel("Gain in model-estimated expected score",labelpad=12)
        ax.xaxis.set_major_locator(MaxNLocator(4));ax.xaxis.set_major_formatter(FuncFormatter(lambda x,pos:f"{x:.3f}"))
    footer(fig,"256 development decision roots · both evaluators shown · fixed-player Maia-3 interval is positive\nExpected-score estimates are model-based; these are not observed human win-rate changes.")
    save(fig,"decision_gain")


def conversion():
    fig,axes=plt.subplots(1,2,figsize=(13,6.6));fig.subplots_adjust(left=.1,right=.96,bottom=.20,top=.70,wspace=.28)
    header(fig,"06 / Prediction to decision","Better prediction is the start of the decision question",
           "Each dot is one development player. Identical scales show the two evaluator families side by side.")
    rows=DATA["prediction_to_decision"]
    for ax,family,color in zip(axes,("E3","E2"),(BLUE,TEAL)):
        x=[r["recorded_NLL_gain"] for r in rows];y=[r[family+"_G"] for r in rows]
        ax.scatter(x,y,s=65,color=color,zorder=3)
        offsets={"P006":(-20,9),"P007":(-30,9),"P013":(5,8),"P014":(7,9),
                 "P025":(-35,-17),"P029":(6,-18),"P030":(6,-17),"P032":(5,9)}
        for r in rows:ax.annotate(r["player_id"],(r["recorded_NLL_gain"],r[family+"_G"]),
            xytext=offsets[r["player_id"]],textcoords="offset points",fontsize=9,color=INK)
        ax.axhline(0,color=MUTED,ls="--",lw=1);ax.set_xlim(0,.145);ax.set_ylim(-.0025,.0042)
        ax.set_title("Maia-3 evaluator" if family=="E3" else "Maia-2 evaluator",loc="left",weight="bold",fontsize=13)
        ax.grid(color=GRID,alpha=.6);ax.set_xlabel("Prediction improvement (NLL reduction)")
        ax.set_ylabel("Model-estimated decision gain")
    footer(fig,"Descriptive eight-player comparison. All prediction gains are positive; decision gains vary by player and evaluator.")
    save(fig,"prediction_to_decision")


if __name__=="__main__":
    for function in (hero,prediction,identity,training,lora,decisions,conversion):function()
    print("Rendered seven figures in SVG and PNG from evidence/results.json.")
