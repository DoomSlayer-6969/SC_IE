# SC_IE

Python replication of Rahiman, Majidi & Khorashadizadeh (2026), *Secure communication and
image encryption using emotional learning-based chaos synchronization*, Multimedia Tools and
Applications 85:680.

## Layout

```
sc_ie/          library code (config, systems, bases, controller, observer, cipher, security)
experiments/    one script per paper result (Figs 6-21, Tables 2-5)
tests/          unit tests
data/input_images/peppers.png   test image (user-supplied)
results/        generated figures, metrics and REPORT.md (paper vs replicated)
```

Every parameter in `sc_ie/config.py` is tagged `[paper]` or `[assumed]`.

## Usage

```
make install-dev
make check      # lint + tests
make run        # run all experiments
```
