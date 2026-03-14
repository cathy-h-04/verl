"""Shared line-plot styling for task comparison figures."""

DATASET_COLORS = {
    "gsm8k": "#295894",
    "rlhf-ff": "#D04A1C",
    "full-hh-rlhf": "#D04A1C",
}

DATASET_LINESTYLES = {
    "gsm8k": "-",
    "rlhf-ff": "--",
    "full-hh-rlhf": "--",
}

DATASET_ALPHAS = {
    "gsm8k": 0.55,
    "rlhf-ff": 0.98,
    "full-hh-rlhf": 0.98,
}

DATASET_MARKERS = {
    "gsm8k": "o",
    "rlhf-ff": "o",
    "full-hh-rlhf": "o",
}
