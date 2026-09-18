"""
One bar chart of FINAL test accuracy across every run, logged back to wandb.

Each training run writes its own `test_acc` to its own wandb summary, and
nothing in wandb puts those side by side on its own: a `wandb.plot.bar` logged
from inside a run can only ever see that one run. So this pulls the finished
runs with the public API and logs a single chart into one stable summary run,
which is overwritten in place on every invocation -- re-run it whenever a model
finishes and the panel grows to include it, instead of accumulating one chart
per invocation.

The number plotted is the summary `test_acc`: the best-val checkpoint measured
once at the end, NOT the last point of the per-epoch test curve. Those differ
whenever the last epoch is not the best one, and the checkpoint number is the
one that was selected without looking at test.

    python moving_mnist/report_test_accuracy.py
    python moving_mnist/report_test_accuracy.py --filter cf_cls_melstm
    python moving_mnist/report_test_accuracy.py --group head_channels

`--group` additionally emits a chart keyed by a config field rather than by run
name, which is what makes a capacity sweep readable (head_channels=2/4/16/64 on
one axis) once several arms exist.

Needs only the wandb client and network access; no torch, so it runs anywhere,
including on a laptop while the cluster is still training.
"""
import argparse

import wandb

# Columns pulled off each run: (table column, where it lives, key).
SUMMARY_FIELDS = [("test_acc", "summary", "test_acc"),
                  ("test_loss", "summary", "test_loss"),
                  ("best_val_acc", "summary", "best_val_acc")]
CONFIG_FIELDS = [("model", "config", "model"),
                 ("head_channels", "config", "head_channels"),
                 ("velocity_pool", "config", "velocity_pool"),
                 ("trained_params", "config", "trained")]


def collect(api, path, name_filter, exclude, metric):
    """Finished runs carrying `metric`, newest first, as plain dicts."""
    rows = []
    for run in api.runs(path):
        if run.name == exclude or (name_filter and name_filter not in run.name):
            continue
        value = run.summary.get(metric)
        if value is None:            # still training, crashed, or a different task
            continue
        row = {"run": run.name, "state": run.state}
        for col, where, key in SUMMARY_FIELDS + CONFIG_FIELDS:
            src = run.summary if where == "summary" else run.config
            row[col] = src.get(key)
        rows.append(row)
    return rows


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--project', default='FERNN-common-fate')
    p.add_argument('--entity', default=None)
    p.add_argument('--filter', default='cf_cls',
                   help='Only runs whose NAME contains this substring. The default keeps '
                        'the classification runs and drops the prediction ones sharing '
                        'the project.')
    p.add_argument('--metric', default='test_acc',
                   help='Summary key to plot. A run without it is skipped, which is how '
                        'in-flight runs stay out of the chart.')
    p.add_argument('--group', default=None,
                   help='Also chart the metric against this CONFIG field instead of the '
                        'run name (e.g. head_channels, model, velocity_pool).')
    p.add_argument('--summary_run', default='cf_cls_test_summary',
                   help='Name/id of the single run these charts are written to. Fixed on '
                        'purpose: re-running overwrites the panel rather than adding one.')
    p.add_argument('--wandb_dir', default='./tmp/')
    p.add_argument('--dry_run', action='store_true',
                   help='Print the table and log nothing.')
    args = p.parse_args()

    api = wandb.Api()
    path = f"{args.entity}/{args.project}" if args.entity else args.project
    rows = collect(api, path, args.filter, args.summary_run, args.metric)
    if not rows:
        raise SystemExit(f"no runs in {path} matching {args.filter!r} carry "
                         f"a summary {args.metric!r} yet")
    rows.sort(key=lambda r: r[args.metric], reverse=True)

    width = max(len(r["run"]) for r in rows)
    num = lambda v, w: f"{v:>{w}.4f}" if isinstance(v, (int, float)) else f"{'-':>{w}}"
    print(f"{'run':<{width}}  {args.metric:>9}  test_loss  best_val  params")
    for r in rows:
        print(f"{r['run']:<{width}}  {num(r[args.metric], 9)}  {num(r['test_loss'], 9)}  "
              f"{num(r['best_val_acc'], 8)}  {r['trained_params'] or '-'}")
    if args.dry_run:
        return

    columns = ["run", "state"] + [c for c, _, _ in SUMMARY_FIELDS + CONFIG_FIELDS]
    table = wandb.Table(columns=columns,
                        data=[[r[c] for c in columns] for r in rows])

    # resume="allow" on a fixed id: the same run is reopened and rewritten, so the
    # project keeps exactly one comparison panel however often this is run.
    wandb.init(project=args.project, entity=args.entity, dir=args.wandb_dir,
               id=args.summary_run, name=args.summary_run, resume="allow",
               job_type="report", config={"n_runs": len(rows), "metric": args.metric})
    payload = {
        f"{args.metric}_by_run": wandb.plot.bar(
            table, "run", args.metric,
            title=f"Final {args.metric} by run ({len(rows)} models)"),
        "test_summary": table,
    }
    if args.group:
        # One bar per distinct value of the config field, averaged over the runs
        # sharing it -- with one seed per arm this is just a relabelling, and with
        # several it is the number the sweep is actually about.
        agg = {}
        for r in rows:
            agg.setdefault(str(r[args.group]), []).append(r[args.metric])
        grouped = wandb.Table(
            columns=[args.group, args.metric, "n_runs"],
            data=sorted(([k, sum(v) / len(v), len(v)] for k, v in agg.items()),
                        key=lambda x: x[1], reverse=True))
        payload[f"{args.metric}_by_{args.group}"] = wandb.plot.bar(
            grouped, args.group, args.metric,
            title=f"Final {args.metric} by {args.group}")
    wandb.log(payload)
    wandb.finish()
    print(f"\nlogged {len(rows)} runs to the '{args.summary_run}' run in {path}")


if __name__ == "__main__":
    main()
