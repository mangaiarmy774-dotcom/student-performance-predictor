# Student Performance Prediction Using Decision Tree Classification

Small Streamlit website that predicts **Likely to Pass** or **Needs Support** from study time,
past failures, absences and the first two period grades.

Dataset: UCI Student Performance (Maths course), saved as `student_data.csv` (semicolon separated).

## Run it
```
pip install -r requirements.txt
python train_model.py      # trains, prints accuracy, saves model + charts
streamlit run app.py       # opens the website
```

## Files
- `train_model.py` - data prep, training, evaluation
- `app.py` - website
- `model.joblib`, `metrics.json` - created by training
- `screenshots/` - confusion matrix, tree diagram (add website screenshots here)

Predictions are estimates, not guarantees.
