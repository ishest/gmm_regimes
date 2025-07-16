# gmm_regimes
This repository contains the complete code, data pipeline, and documentation for the Master’s Final Work: “Mechanism for Identifying Market Regimes Based on a Gaussian Mixture Model” by Igor Shestopalov (2025).

gmm_regimes/
├── data/
│   └── spy_mwf.csv
├── src/
│   ├── data_loader.py
│   ├── feature_engineer.py
│   ├── regime_modeler.py
│   ├── visualizer.py
│   └── main.py
├── requirements.txt
└── README.md


# GMM-Based Market Regime Detection

This project implements an **object-oriented pipeline** for identifying market regimes in financial time series using **Gaussian Mixture Models (GMMs)**. The core use case is detecting "bull," "neutral," and "bear" regimes from daily S&P 500 ETF (SPY) data.

## Structure

- `src/`: Source code
    - `data_loader.py`: Loads CSV data
    - `feature_engineer.py`: Computes features and normalization
    - `regime_modeler.py`: Fits GMM, selects optimal regime count
    - `visualizer.py`: Plots results
    - `main.py`: Runs the full pipeline
- `data/`: Place your CSV file (e.g., `spy_mwf.csv`) here

## Usage

1. **Install dependencies:**
    ```bash
    pip install -r requirements.txt
    ```
2. **Add your data:**  
   Place your SPY data CSV (`spy_mwf.csv`) in the `data/` directory. The CSV must include at least columns: `Date` and `Adj Close`.

3. **Run the main script:**
    ```bash
    cd src
    python main.py
    ```

4. **Outputs:**
    - Plots (BIC curve, scatter, regime probabilities)
    - Regime-labeled CSV (`data/spy_mwf_regimes.csv`)
    - Prints best regime count and model parameters

## Features

- Modular, object-oriented design
- Robust, fully reproducible workflow
- Clear separation of data loading, processing, modeling, and visualization
- Suitable for extension to other assets or additional features

## Citation

If you use this code, please cite:

> Shestopalov, I. (2025). Mechanism for Identifying Market Regimes Based on a Gaussian Mixture Model. MFW, ISEG.

---

## License

[MIT License](LICENSE)
