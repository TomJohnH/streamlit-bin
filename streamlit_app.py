import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import seaborn as sns
import matplotlib.pyplot as plt
import statsmodels.api as sm
from optbinning import OptimalBinning

from optbinning import ContinuousOptimalBinning

st.set_option("deprecation.showPyplotGlobalUse", False)
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

    # https://gnpalencia.org/optbinning/tutorials/tutorial_continuous.html

    tab0, tab1, tab2, tab3 = st.tabs(["Histograms", "Binning", "Correlations", "GLM"])

    df = pd.read_csv(uploaded_file)

    # Select only numerical columns
    cols_names = df.columns.tolist()

    with tab1:
        # Select column to bin
        numerical_cols = df.select_dtypes(include=["float", "int"]).columns.tolist()
        selected_column = st.selectbox("Select column to bin", numerical_cols)

        # Get the target column
        target_column = st.selectbox("Select target column", numerical_cols)

        # Perform optimal binning
        if st.button("Perform Optimal Binning"):
            optb = ContinuousOptimalBinning(name=selected_column, dtype="numerical")
            optb.fit(df[selected_column], df[target_column])
            st.write("Splits")
            st.write(optb.splits)
            st.write("Binning table")
            binning_table = optb.binning_table
            st.write(optb.binning_table.build())
            fig, ax = plt.subplots()
            fig = binning_table.plot()
            st.pyplot(fig)
    # col1, col2 = st.columns(2)

    with tab0:
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

        numerical_cols = df.select_dtypes(include=["float", "int"]).columns.tolist()
        # Add multiselect widget
        response = st.multiselect(
            "Response variable",
            numerical_cols,
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
            df_model = df_model.dropna()
            df_model_inital = df_model

            dtype = df_model[explanatory[0]].dtype

            if dtype == "object":
                st.write(f"{explanatory[0]} is of string data type.")
                # Create dummy variables for the 'x1' categorical variable
                dummies = pd.get_dummies(
                    df_model[explanatory[0]], prefix=explanatory[0], drop_first=True
                )
                df_model = pd.concat([df_model, dummies], axis=1)
                data = df_model.drop(explanatory[0], axis=1)
                explanatory_with_constant = sm.add_constant(
                    df_model.drop([response[0], explanatory[0]], axis=1)
                )
            elif pd.api.types.is_categorical_dtype(dtype):
                st.write(f"{explanatory[0]} is of categorical data type.")
                dummies = pd.get_dummies(
                    df_model[explanatory[0]], prefix=explanatory[0], drop_first=True
                )
                df_model = pd.concat([df_model, dummies], axis=1)
                data = df_model.drop(explanatory[0], axis=1)
                explanatory_with_constant = sm.add_constant(
                    df_model.drop([response[0], explanatory[0]], axis=1)
                )
            else:
                st.write(f"{explanatory[0]} is of other data type.")
                explanatory_with_constant = sm.add_constant(df_model[explanatory[0]])

            # without null records
            st.write("Simple GLM")

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
