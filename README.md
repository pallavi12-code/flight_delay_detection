# ✈️ Flight Delay Prediction (LSTM + Static Features)

A deep learning model that predicts whether a flight will be delayed by
more than 15 minutes, combining a **sequence model (LSTM)** over recent
flight/weather records with a **dense branch** over categorical flight
metadata (carrier, origin, destination).

## Overview

- **Task:** Binary classification — `DELAYED` vs `ON TIME` (delay > 15 min)
- **Data:** [Historical Flight and Weather Data](https://www.kaggle.com/datasets/ioanagheorghiu/historical-flight-and-weather-data) (Kaggle)
- **Architecture:** Dual-input Keras model — a 2-layer LSTM over sequential
  numeric/weather features, concatenated with a dense branch over
  label-encoded categorical features
- **Class imbalance handling:** Focal loss + computed class weights (delays
  are the minority class)
- **Threshold selection:** Chosen from the validation ROC curve (Youden's J
  statistic) rather than a fixed 0.5 cutoff
- **Result:** ~71% prediction accuracy on held-out data

## Project structure

```
flight-delay-prediction/
├── README.md
├── requirements.txt
├── .gitignore
├── data/                    # raw CSVs go here (gitignored — see Setup)
├── models/                  # saved model, encoders, scaler (gitignored)
├── outputs/                 # generated plots (gitignored)
├── notebooks/
│   └── flight_delay_eda_original.py   # original exploratory Colab script
└── src/
    ├── config.py             # paths, feature lists, hyperparameters
    ├── data_preprocessing.py # loading, cleaning, encoding, sequence building
    ├── model.py               # model architecture + focal loss
    ├── train.py                # end-to-end training pipeline
    ├── visualize.py            # confusion matrix / training curve plots
    └── predict.py               # CLI inference on a single flight
```

## Setup

```bash
git clone https://github.com/<your-username>/flight-delay-prediction.git
cd flight-delay-prediction
pip install -r requirements.txt
```

### Get the data

1. Download `kaggle.json` from your Kaggle account (Account → API → Create New Token).
2. ```bash
   mkdir -p ~/.kaggle
   mv kaggle.json ~/.kaggle/
   chmod 600 ~/.kaggle/kaggle.json
   kaggle datasets download -d ioanagheorghiu/historical-flight-and-weather-data
   unzip historical-flight-and-weather-data.zip -d data/flight_weather_data
   ```

## Usage

**Train the model:**
```bash
python -m src.train
```
This cleans the data, builds LSTM sequences, trains with early stopping,
picks an optimal decision threshold on the validation set, and saves the
model, label encoders, scaler, and threshold to `models/`. Plots are saved
to `outputs/`.

**Run inference on a single flight:**
```bash
python -m src.predict
```
Prompts for flight/weather details and returns a delay probability using
the exact encoders/scaler fit during training (no placeholder values).

## Notes on modeling choices

- **Focal loss over plain binary cross-entropy** — with delays as the
  minority class, focal loss keeps the model focused on the harder,
  under-represented examples instead of being dominated by easy negatives.
- **Threshold tuning** — the default 0.5 cutoff isn't necessarily optimal
  under class imbalance, so the operating threshold is chosen from the
  validation ROC curve and reused consistently at inference time.
- **Saved encoders/scaler** — persisting the exact `LabelEncoder`/
  `StandardScaler` objects fit during training (rather than re-fitting or
  using placeholder values at inference) keeps train/serve transforms
  consistent, and unseen categories at inference time fall back to a safe
  default instead of raising an error.

## Possible next steps

- Compare against a non-sequential baseline (e.g. gradient-boosted trees)
  to quantify what the LSTM sequence structure actually adds
- Track experiments (e.g. MLflow/Weights & Biases) instead of print statements
- Add unit tests for `data_preprocessing.py` and a small sample dataset for CI

## License

MIT — see [LICENSE](LICENSE).
