import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import seaborn as sns
import matplotlib.pyplot as plt
import statsmodels.api as sm

# -------------------------
#
#       INTRO
#
# -------------------------

st.title("Binning tool")

# allow user to upload a CSV file containing stock data
uploaded_file = st.file_uploader(
    "Upload data *.csv",
    type=["csv"],
    help="Upload data",
)

# allow user to play around with the app without uploading the file
if st.checkbox("Use example file"):
    uploaded_file = "data.csv"

if uploaded_file is not None:

    tab1, tab2, tab3 = st.tabs(["Histograms", "Correlations", "GLM"])

    df = pd.read_csv(uploaded_file)

    # Select only numerical columns
    cols_names = df.columns.tolist()

    # col1, col2 = st.columns(2)

    with tab1:
        selected_column = st.selectbox("Select a column:", options=cols_names)

        st.write(df[selected_column].head(5))

        # Create histogram
        fig = px.histogram(df, x=selected_column)

        # Show plot
        st.plotly_chart(fig, use_container_width=True)

    with tab2:
        # Calculate the correlation matrix
        corr_matrix = df.corr()

        # Create a heatmap using Seaborn
        fig, ax = plt.subplots()
        sns.heatmap(corr_matrix, annot=True, ax=ax)
        st.pyplot(fig)

    with tab3:
        # Add multiselect widget
        response = st.multiselect(
            "Response variable",
            cols_names,
            max_selections=1,
        )

        explanatory = st.multiselect(
            "Explanatory variable",
            cols_names,
            max_selections=1,
        )
        if response and explanatory:

            df_model = df[[response[0], explanatory[0]]]

            # null records
            st.write("Removed null records")
            null_records = df_model[df_model.isnull().any(axis=1)]
            st.write(null_records)

            # without null records
            st.write("Simple GLM")
            df_model = df_model.dropna()

            explanatory_with_constant = sm.add_constant(df_model[explanatory[0]])

            model = sm.GLM(
                df_model[response[0]],
                explanatory_with_constant,
                family=sm.families.Gaussian(),
            ).fit()

            st.write(model.summary())
            fig, ax = plt.subplots()
            y_pred = model.predict(explanatory_with_constant)
            ax.scatter(df_model[explanatory[0]], df_model[response[0]], color="blue")
            ax.plot(df_model[explanatory[0]], y_pred, color="red", linewidth=2)
            st.pyplot(fig)
