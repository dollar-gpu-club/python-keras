# python-keras

1) Clone the repo:
```
$ git clone git@github.com:dollar-gpu-club/python-keras.git
```

2) `cd` into `python-keras` and run:
```
$ pip install -e .
```

3) Import helper methods
```
from dollar_gpu_club import fit, load_and_compile
```

4) Replace `model.fit(...)` with `fit(model, ...)` and `model.compile(...)` with `load_and_compile(model, ...)`.
