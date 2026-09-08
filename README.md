# ✈️ Flight Delay Prediction

A deep-learning system for predicting whether a flight will be delayed by more than 15 minutes using sequential flight/weather features and categorical flight metadata.

## Overview

**Task:** Binary classification — delayed vs. on time  
**Model:** Dual-input Keras architecture with LSTM sequence modeling and a dense metadata branch  
**Dataset:** Historical Flight and Weather Data (Kaggle)  
**Reported held-out accuracy:** ~71%

## Architecture

```text
Recent Flight / Weather Records
              ↓
        2-Layer LSTM
              │
              ├──────────────┐
              │              ↓
Flight Metadata → Dense Branch
              │              │
              └──────┬───────┘
                     ↓
              Feature Fusion
                     ↓
              Delay Probability
```

## Modeling choices

- **Focal loss** to focus learning on harder minority-class delay examples
- **Class weights** to address imbalance
- **Validation-based threshold selection** using Youden's J statistic rather than assuming a 0.5 cutoff
- Persisted encoders and scaler so inference uses the same transformations as training
- Safe handling of unseen categorical values during inference

## Project structure

```text
flight_delay_detection/
├── requirements.txt
├── .gitignore
├── notebooks/
│   └── flight_delay_eda_original.py
└── src/
    ├── config.py
    ├── data_preprocessing.py
    ├── model.py
    ├── train.py
    ├── visualize.py
    └── predict.py
```

## Run locally

```bash
git clone https://github.com/pallavi12-code/flight_delay_detection.git
cd flight_delay_detection
pip install -r requirements.txt
```

Download the dataset from Kaggle and place the extracted data in the expected `data/` location.

Train:

```bash
python -m src.train
```

Run inference:

```bash
python -m src.predict
```

## Tech stack

**Python · TensorFlow/Keras · LSTM · Scikit-learn · Pandas · NumPy**

## Future improvements

- Benchmark the LSTM against tree-based and simpler non-sequential baselines
- Add automated unit tests and a small CI dataset
- Track experiments with MLflow or Weights & Biases
- Report precision, recall, ROC-AUC and calibration alongside accuracy

## License

MIT — see [LICENSE](LICENSE).
