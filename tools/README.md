# Experiment tools

Generate a catalog-held-out synthetic set with the official scenario mix:

```bash
python tools/generate_synthetic.py --count 2000 --seed 20260827
```

Run the public and synthetic gates sequentially into a timestamped `runs/` folder:

```bash
python tools/run_experiments.py --synthetic-count 2000 --seed 20260827
```

Compare two or more evaluator outputs (the first is the reference):

```bash
python tools/compare_results.py baseline.json candidate.json
```

Synthetic targets exclude every public target ASIN. Generated datasets and run outputs are ignored by Git.
