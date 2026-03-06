import streamlit as st
import pandas as pd
import joblib
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
import xgboost as xgb
import os

NUMERIC_FEATURES = [
    'RevolvingUtilizationOfUnsecuredLines', 'age', 'DebtRatio',
    'MonthlyIncome', 'NumberOfOpenCreditLinesAndLoans',
    'NumberRealEstateLoansOrLines', 'NumberOfDependents',
    'TotalDelinquencies'
]

# ── Fallback training statistics (from Give Me Some Credit dataset)
# Used only if saved_imputer.pkl / saved_scaler.pkl are not found
TRAIN_MEDIANS = {
    'RevolvingUtilizationOfUnsecuredLines': 0.1539,
    'age': 52.0,
    'DebtRatio': 0.3665,
    'MonthlyIncome': 5400.0,
    'NumberOfOpenCreditLinesAndLoans': 8.0,
    'NumberRealEstateLoansOrLines': 1.0,
    'NumberOfDependents': 0.0,
    'TotalDelinquencies': 0.0,
}
TRAIN_STD = {
    'RevolvingUtilizationOfUnsecuredLines': 0.4321,
    'age': 14.77,
    'DebtRatio': 2.038,
    'MonthlyIncome': 14384.0,
    'NumberOfOpenCreditLinesAndLoans': 5.15,
    'NumberRealEstateLoansOrLines': 1.16,
    'NumberOfDependents': 1.12,
    'TotalDelinquencies': 1.87,
}


@st.cache_resource
def load_artifacts():
    """Load model, imputer, scaler — cached so loaded only once."""
    # Load model
    model = xgb.XGBClassifier()
    model.load_model('xgb_model.json')

    # Load imputer and scaler if available
    imputer = joblib.load('saved_imputer.pkl') if os.path.exists('saved_imputer.pkl') else None
    scaler  = joblib.load('saved_scaler.pkl')  if os.path.exists('saved_scaler.pkl')  else None

    return model, imputer, scaler


def preprocess_data_csv(test_df, imputer, scaler):
    """
    Preprocess uploaded CSV data.
    Uses saved imputer/scaler fitted on training data.
    """
    test_df = test_df.copy()
    test_df = test_df.drop(['SeriousDlqin2yrs'], axis=1, errors='ignore')

    # Composite feature — same as training notebook
    test_df['TotalDelinquencies'] = (
        test_df['NumberOfTimes90DaysLate'] +
        test_df['NumberOfTime60-89DaysPastDueNotWorse'] +
        test_df['NumberOfTime30-59DaysPastDueNotWorse']
    )

    test_df.drop([
        'NumberOfTimes90DaysLate',
        'NumberOfTime60-89DaysPastDueNotWorse',
        'NumberOfTime30-59DaysPastDueNotWorse'
    ], axis=1, inplace=True, errors='ignore')

    if imputer is not None and scaler is not None:
        # ✅ Use training-fitted objects — correct approach
        test_df[NUMERIC_FEATURES] = imputer.transform(test_df[NUMERIC_FEATURES])
        test_df[NUMERIC_FEATURES] = scaler.transform(test_df[NUMERIC_FEATURES])
    else:
        # ✅ Fallback — apply training medians and stds manually
        for col in NUMERIC_FEATURES:
            test_df[col] = test_df[col].fillna(TRAIN_MEDIANS[col])
        for col in NUMERIC_FEATURES:
            test_df[col] = (test_df[col] - TRAIN_MEDIANS[col]) / TRAIN_STD[col]

    return test_df


def preprocess_data(test_df, imputer, scaler):
    """
    Preprocess single manually entered customer.
    Uses saved imputer/scaler fitted on training data.
    """
    test_df = test_df.copy()

    if imputer is not None and scaler is not None:
        # ✅ Use training-fitted objects — correct approach
        test_df[NUMERIC_FEATURES] = imputer.transform(test_df[NUMERIC_FEATURES])
        test_df[NUMERIC_FEATURES] = scaler.transform(test_df[NUMERIC_FEATURES])
    else:
        # ✅ Fallback — apply training medians and stds manually
        for col in NUMERIC_FEATURES:
            test_df[col] = test_df[col].fillna(TRAIN_MEDIANS[col])
        for col in NUMERIC_FEATURES:
            test_df[col] = (test_df[col] - TRAIN_MEDIANS[col]) / TRAIN_STD[col]

    return test_df


def main():
    st.title('Credit Risk Prediction')

    # Load model and preprocessing artifacts
    model, imputer, scaler = load_artifacts()

    # Warn if pkl files are missing — fallback is active
    if imputer is None or scaler is None:
        st.warning(
            "⚠️ `saved_imputer.pkl` or `saved_scaler.pkl` not found. "
            "Using fallback training statistics. "
            "For best accuracy run `save_artifacts` code in your notebook and add the files here.",
            icon="⚠️"
        )

    # Sidebar input option
    st.sidebar.subheader('Data Input')
    input_option = st.sidebar.radio('Select Input Option', ('Upload CSV', 'Enter Manually'))

    # ── CSV Upload ──────────────────────────────────────
    if input_option == 'Upload CSV':
        uploaded_file = st.sidebar.file_uploader('Upload CSV', type=['csv'])

        if uploaded_file is not None:
            data = pd.read_csv(uploaded_file)
            st.subheader('Uploaded Data')

            if 'Unnamed: 0' in data.columns:
                data.drop(['Unnamed: 0'], axis=1, inplace=True)
            st.write(data)

            # Preprocess
            preprocessed_data = preprocess_data_csv(data.copy(), imputer, scaler)

            # Predict
            prediction       = model.predict(preprocessed_data)
            prediction_proba = model.predict_proba(preprocessed_data)[:, 1]

            # Build results DataFrame
            predictions_df = data.copy()
            predictions_df['Predicted SeriousDlqin2yrs']        = prediction
            predictions_df['Predicted Probability of default']   = prediction_proba

            if 'SeriousDlqin2yrs' in predictions_df.columns:
                predictions_df.drop('SeriousDlqin2yrs', axis=1, inplace=True)

            predictions_df.sort_values(
                'Predicted Probability of default', ascending=False, inplace=True
            )

            cols = ['Predicted SeriousDlqin2yrs', 'Predicted Probability of default'] + \
                   [c for c in predictions_df.columns
                    if c not in ['Predicted SeriousDlqin2yrs', 'Predicted Probability of default']]
            predictions_df = predictions_df[cols]

            st.subheader('Predictions')
            st.write(predictions_df)

    # ── Manual Entry ────────────────────────────────────
    else:
        st.subheader('Enter Customer Information')

        revolving_utilization   = st.number_input('Revolving Utilization of Unsecured Lines', min_value=0.0, value=0.0)
        age                     = st.number_input('Age', min_value=0, value=30)
        debt_ratio              = st.number_input('Debt Ratio', min_value=0.0, value=0.0)
        monthly_income          = st.number_input('Monthly Income', min_value=0.0, value=5000.0)
        open_credit_lines_loans = st.number_input('Number of Open Credit Lines and Loans', min_value=0, value=0)
        real_estate_loans       = st.number_input('Number of Real Estate Loans or Lines', min_value=0, value=0)
        dependents              = st.number_input('Number of Dependents', min_value=0, value=0)
        total_delinquencies   = st.number_input('TotalDelinquencies', min_value=0, value=0)

        if st.button('Predict'):
            data = pd.DataFrame({
                'RevolvingUtilizationOfUnsecuredLines': [revolving_utilization],
                'age':                                  [age],
                'DebtRatio':                            [debt_ratio],
                'MonthlyIncome':                        [monthly_income],
                'NumberOfOpenCreditLinesAndLoans':      [open_credit_lines_loans],
                'NumberRealEstateLoansOrLines':          [real_estate_loans],
                'NumberOfDependents':                   [dependents],
                'TotalDelinquencies':                   [total_delinquencies],
            })

            # Preprocess using training statistics — NOT fit on this single row
            preprocessed_data = preprocess_data(data, imputer, scaler)

            # Predict
            prediction       = model.predict(preprocessed_data)
            prediction_proba = model.predict_proba(preprocessed_data)[:, 1]

            # Display result
            st.subheader('Prediction Result')

            prob = float(prediction_proba[0])
            pred = int(prediction[0])

            # Risk badge
            if prob >= 0.60:
                st.error(f"🔴 HIGH RISK — Probability of Default: {prob:.1%}")
            elif prob >= 0.30:
                st.warning(f"🟡 MEDIUM RISK — Probability of Default: {prob:.1%}")
            else:
                st.success(f"🟢 LOW RISK — Probability of Default: {prob:.1%}")

            st.write(pd.DataFrame({
                'Predicted SeriousDlqin2yrs':      [pred],
                'Predicted Probability of default': [round(prob, 4)]
            }))


if __name__ == '__main__':
    main()
