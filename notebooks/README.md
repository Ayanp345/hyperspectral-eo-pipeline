# Notebooks

every stage of the
pipeline is already exposed as an independent, well-commented CLI stage in
`main.py` (see the root `README.md`, section 7), which is easier to diff,
test, and run headlessly than a notebook, while still being fully runnable
stage-by-stage for exploration:

```bash
python main.py generate      # inspect the simulated scene
python main.py preprocess    # + calibration / atmospheric correction / PCA-MNF
python main.py classical     # + SAM / unmixing / RX / matched filter
python main.py baseline      # + Random Forest
python main.py apps          # + application products
python main.py deep          # + HybridSN / Spectral-Spatial Transformer
```

If you prefer an interactive notebook, the quickest path is to paste the
body of `main.py`'s `cmd_all()` function into a notebook cell — every
function it calls is a plain, side-effect-documented Python function
importable from `src/`.
