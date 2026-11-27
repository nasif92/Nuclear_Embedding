"""Summarise run_compare.sh: every model, every stain, CV (training slides) and reference slides side by side.

    python3 set_transformers/compare_all.py runs/compare [--baseline mlp]

Layout read:  <root>/<stain>/cv/<config>_f<fold>.log    (RESULT with holdout_per_slide)
              <root>/<stain>/ref/<config>_s<seed>.log   (RESULT with per_slide / per_slide_labelled)

Per stain:
  CV      slide-mean over every held-out training slide; paired delta vs the baseline over the same slides
          (mean of per-slide differences) and on how many slides the config is better.
  REF     annotated reference slide-mean (mean +- sd over seeds), labelled-binary slide-mean, and the paired
          same-seed delta vs the baseline with slides better.
Last: one line per config across stains (CV and reference), so a model has to hold up everywhere.
"""
import argparse
import glob
import json
import os
import re
import statistics as st
from collections import defaultdict

ORDER = ["mlp", "mlp_norm", "set1", "set32", "shuf48", "set1_norm", "set32_norm", "shuf48_norm",
         "fus_mlp", "fus_mlp_norm", "fus_set32", "fus_set32_norm",
         "lem_mlp", "lem_shuf48"] + [f"{p}_{k}" for p in ("n2v", "n2vm", "n2vs") for k in ("mlp", "shuf48", "fus_mlp", "fus_mlp_norm", "fus_shuf48_norm")]


def result(path):
    res = None
    for line in open(path, errors="replace"):
        if line.startswith("RESULT "):
            res = json.loads(line[7:])
    return res


def ms(xs):
    if not xs:
        return f"{'-':>15}"
    return f"{st.mean(xs):7.2f} +- {st.stdev(xs):4.2f}" if len(xs) > 1 else f"{xs[0]:7.2f}        "


def load(root):
    cv = defaultdict(lambda: defaultdict(dict))     # stain -> cfg -> slide -> acc
    ref = defaultdict(lambda: defaultdict(dict))    # stain -> cfg -> seed -> result
    bad = []
    for stain_dir in sorted(glob.glob(os.path.join(root, "*"))):
        stain = os.path.basename(stain_dir)
        for p in glob.glob(os.path.join(stain_dir, "cv", "*.log")):
            m = re.match(r"(.+)_f(\d+)\.log$", os.path.basename(p))
            r = result(p) if m else None
            if not r or not r.get("holdout_per_slide"):
                bad.append(p); continue
            cv[stain][m.group(1)].update(r["holdout_per_slide"])
        for p in glob.glob(os.path.join(stain_dir, "ref", "*.log")):
            m = re.match(r"(.+)_s(\d+)\.log$", os.path.basename(p))
            r = result(p) if m else None
            if not r or not r.get("per_slide"):
                bad.append(p); continue
            ref[stain][m.group(1)][int(m.group(2))] = r
    return cv, ref, bad


def order(cfgs):
    return [c for c in ORDER if c in cfgs] + sorted(c for c in cfgs if c not in ORDER)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("root", nargs="?", default="runs/compare")
    ap.add_argument("--baseline", default="mlp")
    ap.add_argument("--per_slide", action="store_true", help="also print per-slide reference accuracy")
    a = ap.parse_args()
    cv, ref, bad = load(a.root)
    if bad:
        print(f"{len(bad)} log(s) without a usable RESULT (running, failed, or not finished), e.g.:")
        for p in sorted(bad)[:8]:
            print("   ", p)
    overall = defaultdict(dict)
    for stain in sorted(set(cv) | set(ref)):
        print(f"\n{'=' * 30} {stain} {'=' * 30}")
        C = cv.get(stain, {})
        if C:
            base = C.get(a.baseline, {})
            n_all = max(len(v) for v in C.values())
            print(f"\n--- CV on training slides (seed 42; complete = {n_all} slides) ---")
            print(f"{'config':<16}{'slides':>7}{'CV slide-mean':>15}{'vs ' + a.baseline:>12}{'better':>10}")
            for c in order(C):
                v = C[c]
                cvm = 100 * st.mean(v.values())
                common = sorted(set(v) & set(base))
                if common and c != a.baseline:
                    d = [100 * (v[s] - base[s]) for s in common]
                    dl = f"{st.mean(d):+12.2f}{sum(x > 0 for x in d):>5}/{len(d):<4}"
                else:
                    dl = f"{'':>12}{'':>10}"
                flag = "" if len(v) == n_all else "  (incomplete)"
                print(f"{c:<16}{len(v):>7}{cvm:>15.2f}{dl}{flag}")
                overall[c][(stain, "cv")] = cvm if len(v) == n_all else None
        R = ref.get(stain, {})
        if R:
            base = R.get(a.baseline, {})
            print(f"\n--- Reference slides (seeds; annotated = pathologist regions, labelled = labelled-binary) ---")
            print(f"{'config':<16}{'seeds':>6}{'annotated':>17}{'vs ' + a.baseline:>12}{'better':>9}{'labelled':>17}{'vs ' + a.baseline:>12}")
            for c in order(R):
                runs = R[c]
                ann = [100 * st.mean(r["per_slide"].values()) for r in runs.values()]
                lab = [100 * st.mean(r["per_slide_labelled"].values()) for r in runs.values() if r.get("per_slide_labelled")]
                seeds = sorted(set(runs) & set(base))
                da = dl_ = better = ""
                if seeds and c != a.baseline:
                    da = f"{st.mean(100 * (st.mean(runs[s]['per_slide'].values()) - st.mean(base[s]['per_slide'].values())) for s in seeds):+12.2f}"
                    slides = sorted(set.intersection(*[set(runs[s]["per_slide"]) & set(base[s]["per_slide"]) for s in seeds]))
                    b = sum(st.mean(runs[s]["per_slide"][sl] - base[s]["per_slide"][sl] for s in seeds) > 0 for sl in slides)
                    better = f"{b:>5}/{len(slides):<3}"
                    ls = [s for s in seeds if runs[s].get("per_slide_labelled") and base[s].get("per_slide_labelled")]
                    if ls:
                        dl_ = f"{st.mean(100 * (st.mean(runs[s]['per_slide_labelled'].values()) - st.mean(base[s]['per_slide_labelled'].values())) for s in ls):+12.2f}"
                print(f"{c:<16}{len(runs):>6}{ms(ann):>17}{da:>12}{better:>9}{ms(lab):>17}{dl_:>12}")
                overall[c][(stain, "ref")] = st.mean(ann)
                overall[c][(stain, "lab")] = st.mean(lab) if lab else None
            if a.per_slide:
                slides = sorted({sl for runs in R.values() for r in runs.values() for sl in r["per_slide"]})
                print(f"\n{'config':<16}" + "".join(f"{sl.split(':')[-1][:12]:>14}" for sl in slides))
                for c in order(R):
                    rs = list(R[c].values())
                    print(f"{c:<16}" + "".join(
                        f"{100 * st.mean(r['per_slide'][sl] for r in rs if sl in r['per_slide']):>14.1f}" for sl in slides))

    stains = sorted(set(cv) | set(ref))
    if len(stains) > 1 or True:
        print(f"\n{'=' * 30} ALL STAINS {'=' * 30}")
        hdr = "".join(f"{s + ' ' + k:>13}" for s in stains for k in ("cv", "ref", "lab"))
        print(f"{'config':<16}{hdr}{'mean cv':>10}{'mean ref':>10}")
        for c in order(overall):
            vals = [overall[c].get((s, k)) for s in stains for k in ("cv", "ref", "lab")]
            cvs = [overall[c].get((s, "cv")) for s in stains]
            rfs = [overall[c].get((s, "ref")) for s in stains]
            mc = f"{st.mean(cvs):10.2f}" if all(x is not None for x in cvs) else f"{'-':>10}"
            mr = f"{st.mean(rfs):10.2f}" if all(x is not None for x in rfs) else f"{'-':>10}"
            print(f"{c:<16}" + "".join(f"{v:13.2f}" if v is not None else f"{'-':>13}" for v in vals) + mc + mr)
        print("\ncv = CV slide-mean on training slides; ref = annotated reference slide-mean; lab = labelled-binary "
              "(fusion configs have none). Mean columns only when every stain is complete.")


if __name__ == "__main__":
    main()