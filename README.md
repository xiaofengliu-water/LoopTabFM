<div align="center">

# LoopTabFM: Looping Tabular Foundation Model with Soft Pseudo-labels and Risk-aware Updates

[![arxiv](https://img.shields.io/static/v1?label=arXiv&message=2410.18164&color=B31B1B&logo=arXiv)](https://arxiv.org/abs/2506.15329)
</div>

This repository contains the official implementation of **LoopTabFM** proposed in our paper:

> **[When and How Unlabeled Data Provably Improve In-Context Learning](https://arxiv.org/abs/2506.15329)**
> 
> *Yingcong Li, Xiangyu Chang, Muti Kara, Xiaofeng Liu, Amit Roy-Chowdhury, Samet Oymak*

---

## Overview
LoopTabFM is a semi-supervised learning framework designed to enhance predictive performance on tabular datasets with extremely limited labeled data. This algorithm iteratively creates soft pseudo-labels by explicitly looping the tabular FM while controlling validation risk. Focusing on the few-shot learning setting where TabPFN-v2 (Hollmann et al., 2025) excels, we demonstrate that our
approach can significantly improve predictive performance on various real-world datasets.


## Theory & Mechanism

The algorithm is grounded in the theoretical finding that while single-layer models often fail to exploit unlabeled data, multilayer or **looped transformers** can provably leverage them to reach Bayes-optimal performance.

### Key Components:
* **Soft-Label Mapping:** Instead of hard classification, we map labels to a continuous space $\{-1, 1\}$ and use `TabPFNRegressor` to preserve prediction uncertainty.
* **Looping Inference:** The model state is iteratively updated by incorporating the most "decisive" unlabeled samples back into the context.
* **Proxy Validation ($V_{proxy}$):** To prevent semantic drift without a labeled validation set, we monitor the decisiveness of predictions on the unlabeled pool:
    $$V_{proxy} = \frac{1}{n} \sum \min(|y_{pseudo} - 1|, |y_{pseudo} + 1|)$$

---

## Project Structure

```text
├── data/
│   └── Evaluated_Datasets.csv                    # OpenML dataset IDs 
├── output/                                       # Generated accuracy reports and plots
├── looptabfm.py                                  # Main algorithm 
├── requirements.txt                              # Dependencies
└── README.md                                     # Documentation
```

## Installation

To reproduce the results from the paper, you must use **TabPFN version 2.0.9**.

1. **Clone the repository:**
   ```bash
   git clone [https://github.com/xiaofengliu-water/LoopTabFM.git](https://github.com/xiaofengliu-water/LoopTabFM.git)
   cd LoopTabFM

2. **Create a virtual environment (Recommended):**
   ```bash
   python -m venv venv
   # On Mac/Linux:
   source venv/bin/activate
   # On Windows:
   .\venv\Scripts\activate

3. **Install dependencies:**
   ```bash
   pip install -r requirements.txt

## Usage
To run the experiment across the datasets defined in the `data/` folder:

```python
python looptabfm.py
```

### Configuration

You can adjust the experimental settings by modifying the following parameters in `looptabfm.py`:

| Parameter | Default | Description |
| :--- | :---: | :--- |
| `n_tt_splits` | `100` | Number of random train-test trials for averaging the results. |
| `n_init_lab` | `10` | Size of the initial labeled dataset. |
| `n_init_unlab` | `10` | Size of the unlabeled dataset. |
| `n_iters` | `5` | Number of loops (iterations). |


## Results & Visualization
The script automatically generates two types of outputs in the `output/` folder:

Excel Reports (`.xlsx`): Containing mean accuracy and proxy validation scores across all iterations.

Accuracy Plots (`.png`): Visualizing the performance trajectory as the model "loops" through the unlabeled data.

## Citation

If you use this LoopTabFM algorithm in your research, please cite:

```
@article{li2025,
  title={When and How Unlabeled Data Provably Improve In-Context Learning},
  author={Li, Yingcong and Chang, Xiangyu and Kara, Muti and Liu, Xiaofeng and Roy-Chowdhury, Amit and Oymak, Samet},
  journal={arXiv preprint arXiv:2506.15329},
  year={2025}
}
```
