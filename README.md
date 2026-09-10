# Flight Delay Prediction

This repository contains an LSTM-based binary classifier for whether a flight's
arrival delay exceeds 15 minutes. It is an experiment, not a production-ready
forecasting service, and this repository does not contain the source dataset or
trained artifacts.

## Architecture and data flow

```text
CSV files
  -> cleaning and target creation
  -> chronological 60/20/20 split
  -> fit encoders/scaler on training rows only
  -> preceding-row windows (T=5)
  -> LSTM sequence branch + categorical metadata branch
  -> focal-loss training
  -> validation threshold selection
  -> held-out test metrics and saved artifacts
```

The model keeps the original two-branch architecture: two LSTM layers process
the numeric history, a dense branch processes encoded carrier/origin/
destination values, and the branches are fused for a sigmoid probability.
The implementation does not add embeddings or change the layer sizes.

## Dataset

Download the Historical Flight and Weather Data dataset separately and place
its monthly CSV files in `data/flight_weather_data/`, or pass another directory
with `--data-dir`. CSV files are loaded in sorted filename order. Required
columns are the features listed in `config.py` plus `arrival_delay`.

## Installation and commands

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Train with configurable paths and hyperparameters:

```bash
python train.py --data-dir data/flight_weather_data \
  --model-dir models --output-dir outputs \
  --seed 42 --epochs 30 --batch-size 256
```

The training command writes the Keras model, label encoders, scaler, selected
threshold, metadata, metrics, and plots to the configured directories.

Run interactive prediction from those artifacts:

```bash
python predict.py --model-dir models
```

The same persisted preprocessing artifacts are used during prediction.
Unseen categorical values are mapped to a known fallback category; missing
inputs and missing artifacts raise explicit errors.

## Testing and CI

The preprocessing and utility tests do not require the full dataset or a
trained TensorFlow model:

```bash
pytest -q
```

GitHub Actions runs these tests on pushes and pull requests. TensorFlow is
intentionally not installed by the lightweight CI job because the tests do not
import or execute the model.

## Leakage audit and limitations

- The scaler and categorical encoders are fit only on the chronological
  training partition. Validation and test rows are transformed with those
  artifacts.
- Evaluation is performed on the final chronological partition, not on rows
  used for fitting. The validation partition is used to choose the operating
  threshold.
- The source data is loaded in filename order; this is only a valid temporal
  ordering if the dataset filenames encode chronology. A real deployment should
  split using a verified flight timestamp and should group related flights as
  appropriate.
- `departure_delay` and the component delay fields may only be known after
  operations have begun. If the intended use is pre-departure prediction,
  these fields are target leakage for that use case and must be removed.
- Numeric missing values are imputed with medians fit on the training
  partition, and those medians are persisted with the preprocessing artifacts.
- Single-flight prediction repeats the current numeric row five times because
  no history is supplied. This is a documented approximation, not equivalent
  to a true rolling-flight inference path.
- No performance metrics are claimed in this README. Results depend on the
  downloaded dataset, split ordering, hardware, and runtime versions.

## Repository layout

```text
config.py              Runtime defaults and artifact paths
data_preprocessing.py  Loading, cleaning, fitted transforms, sequences
model.py               Original LSTM + metadata model
train.py               Chronological training entry point
evaluation.py          Held-out evaluation utility
predict.py             Artifact-backed interactive prediction
visualize.py           Headless plots
tests/                 Dataset-free preprocessing tests
```

## License

MIT — see [LICENSE](LICENSE).
