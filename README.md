# ⚡ PokéClassifier — Legendary Detection Engine

A modern Streamlit GUI for the Pokémon Legendary Classification ML project.

## Setup

```bash
pip install -r requirements.txt
streamlit run app.py
```

## Usage

1. **Upload** `pokemon_complete.csv` using the sidebar uploader.
2. **Explore** your data in the "Explore Data" tab — class balance, type distributions, stat comparisons, and correlation heatmap.
3. **Train** models from the sidebar: choose which models to include, adjust the test split %, and toggle SMOTE on/off. Hit **🚀 Train Models**.
4. **Compare** recall, precision, F1, accuracy, and confusion matrices in the "Train & Compare" tab.
5. **Predict** whether a custom Pokémon is legendary in the "Predict Legendary" tab — dial in stats, choose a model, and see the verdict.

## Models Included
- Logistic Regression
- K-Nearest Neighbours
- Support Vector Machine (RBF kernel)
- Random Forest
- Decision Tree

## Features Used
`type_1`, `type_2`, `hp`, `attack`, `defense`, `sp_attack`, `sp_defense`, `speed`, `base_stat_total`, `base_experience`, `capture_rate`, `is_baby`, `is_mythical`

Target: `is_legendary`
