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


@st.cache_data
def convert_df(df):
    # IMPORTANT: Cache the conversion to prevent computation on every rerun
    return df.to_csv(index=False).encode("utf-8")


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

    tab0, tab1, tab2, tab3 = st.tabs(
        ["Histograms", "Continous binning", "Correlations", "GLM"]
    )

    df = pd.read_csv(uploaded_file)

    # Select only numerical columns
    cols_names = df.columns.tolist()

    with tab0:

        st.subheader("Histogram")
        selected_column = st.selectbox("Select a column:", options=cols_names)

        dtype = df[selected_column].dtype

        if dtype != "object":

            # Define min and max ages
            min_var = int(df[selected_column].min())
            max_var = int(df[selected_column].max())

            # Add slider
            var_range = st.slider(
                "Select age range", min_var, max_var, (min_var, max_var)
            )

            # Filter DataFrame
            filtered_df = df[
                (df[selected_column] >= var_range[0])
                & (df[selected_column] <= var_range[1])
            ]

        # Create histogram
        if dtype == "object":
            fig = px.histogram(df, x=selected_column)
        else:
            fig = px.histogram(filtered_df, x=selected_column)

        # Show plot
        st.plotly_chart(fig, use_container_width=True)
        st.subheader("Unique rows")
        # show head
        st.write("Unique rows")

        st.write(df[selected_column].drop_duplicates())

        csv = convert_df(df[selected_column].drop_duplicates())

        st.download_button(
            label="Download data as CSV",
            data=csv,
            file_name=f"unique_df_{selected_column}.csv",
            mime="text/csv",
        )

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
            st.subheader("Splits")
            st.write(optb.splits)
            st.subheader("Binning table")
            binning_table = optb.binning_table
            st.write(optb.binning_table.build())
            fig, ax = plt.subplots()
            fig = binning_table.plot()
            st.subheader("Binning chart")
            st.pyplot(fig)

            # Get bin edges
            bin_edges = optb.splits

            # Convert bin edges into bin indices using numpy.digitize
            bin_indices = np.digitize(df[selected_column], bins=bin_edges)

            # Create a new column with binned values (bin indices)
            df["binned_indices_" + selected_column] = bin_indices

            # Get the binning table and extract bin intervals
            binning_table_df = binning_table.build()
            bin_intervals = binning_table_df["Bin"].tolist()

            bin_index_to_name = {i: bin_intervals[i] for i in range(len(bin_intervals))}

            # Create custom bin names based on intervals
            df["binned_names_" + selected_column] = df[
                "binned_indices_" + selected_column
            ].map(bin_index_to_name)
            st.subheader("Binned data")
            st.write(df)

            csv = convert_df(df)

            st.download_button(
                label="Download data as CSV",
                data=csv,
                file_name=f"binned_df_{selected_column}.csv",
                mime="text/csv",
            )

    # col1, col2 = st.columns(2)

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
