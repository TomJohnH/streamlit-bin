import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import seaborn as sns
import matplotlib.pyplot as plt
import statsmodels.api as sm
from pygam import LinearGAM, s
from optbinning import OptimalBinning
from optbinning import ContinuousOptimalBinning
# from streamlit_pandas_profiling import st_profile_report
# import pandas_profiling

# turn off graphs warnings
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

# Allow user to upload a CSV file containg the data
uploaded_file = st.file_uploader(
    "Upload data *.csv",
    type=["csv"],
    help="Upload data",
)

# Allow user to play around with the app without uploading the file
if st.checkbox("Use example file"):
    uploaded_file = "data.csv"

if uploaded_file is not None:

    # Split tool to tabs
    tab0, tab1, tab2, tab3, tab4 = st.tabs(
        ["Histograms", "Continous binning", "Correlations", "GLM","Pandas profiler"]
    )

    # Read dataframe
    df = pd.read_csv(uploaded_file)
    df_profiling = df
    # Select columnsnames
    cols_names = df.columns.tolist()

    with tab0:

        st.subheader("Histogram")

        # Allow user to select column for the histogram
        selected_column = st.selectbox("Select a column:", options=cols_names)

        # Check column type
        dtype = df[selected_column].dtype

        # Display slider only if the column is numercial
        if dtype != "object":

            # Define min and max variable values
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

        # Create table with unqie rows
        st.subheader("Unique rows")
        # show head
        st.write("Unique rows")

        st.write(df[selected_column].drop_duplicates())

        # Allow user to download unique column values
        csv = convert_df(df[selected_column].drop_duplicates())

        st.download_button(
            label="Download data as CSV",
            data=csv,
            file_name=f"unique_df_{selected_column}.csv",
            mime="text/csv",
        )

    with tab1:

        # Binning can be extremaly important for modelling
        # this section of the tool aims to simplify binning process

        # https://gnpalencia.org/optbinning/tutorials/tutorial_continuous.html
        with st.expander("Intro to optimal binning with continuous target"):
            st.write(
                """Optimal binning, also known as supervised discretization, is a technique used in the preprocessing stage of machine learning, particularly for classification and regression problems. It involves converting continuous features into discrete intervals or bins, while taking into consideration the relationship between the features and a continuous target variable. The goal is to maximize the information value or predictive power of the binned features.
                
The optbinning package in Python is a popular library for optimal binning. When working with a continuous target variable, the library provides the ContinuousOptimalBinning class to perform the optimal binning process.

The main goal of the optimal binning algorithm with continuous target variables is to maximize the relationship between the binned feature and the target variable. For continuous targets, the algorithm typically aims to maximize the homogeneity within the bins concerning the target variable. In this context, homogeneity means that the values within each bin are as similar as possible.

One common approach to achieve this is by minimizing the within-bin variance or sum of squared differences between the target variable and the mean value of the target variable within each bin. This can be seen as a variation of the decision tree algorithm, where the method tries to find the best split points in the feature space to create the most homogeneous bins.
           
            """
            )
            st.write(
                "https://gnpalencia.org/optbinning/tutorials/tutorial_continuous.html"
            )
        # Select column to bin
        numerical_cols = df.select_dtypes(include=["float", "int"]).columns.tolist()
        selected_column = st.selectbox("Select column to bin:", numerical_cols)

        # Get the target column
        target_column = st.selectbox("Bin based on target:", numerical_cols)

        # Perform optimal binning
        if st.button("Perform Optimal Continous Binning"):

            optb = ContinuousOptimalBinning(name=selected_column, dtype="numerical")
            optb.fit(df[selected_column], df[target_column])

            # Show splits
            st.subheader("Splits")
            st.write(optb.splits)

            # Show binning table
            st.subheader("Binning table")
            binning_table = optb.binning_table
            st.write(optb.binning_table.build())

            # Show binning charts
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

            # Download file
            csv = convert_df(df)

            st.download_button(
                label="Download data as CSV",
                data=csv,
                file_name=f"binned_df_{selected_column}.csv",
                mime="text/csv",
            )

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
        response = st.selectbox(
            "Response variable",
            numerical_cols,
        )

        explanatory = st.selectbox(
            "Explanatory variable",
            cols_names,
        )
        if response and explanatory and response != explanatory:

            df_model = df[[response, explanatory]]

            # null records
            st.write("Removed null records")
            null_records = df_model[df_model.isnull().any(axis=1)]
            st.write(null_records)
            df_model = df_model.dropna()
            df_model_inital = df_model

            dtype = df_model[explanatory].dtype

            if dtype == "object":
                st.write(f"{explanatory} is of string data type.")
                # Create dummy variables for the 'x1' categorical variable
                dummies = pd.get_dummies(
                    df_model[explanatory], prefix=explanatory, drop_first=True
                )
                df_model = pd.concat([df_model, dummies], axis=1)
                data = df_model.drop(explanatory, axis=1)
                explanatory_with_constant = sm.add_constant(
                    df_model.drop([response, explanatory], axis=1)
                )
            elif pd.api.types.is_categorical_dtype(dtype):
                st.write(f"{explanatory} is of categorical data type.")
                dummies = pd.get_dummies(
                    df_model[explanatory], prefix=explanatory, drop_first=True
                )
                df_model = pd.concat([df_model, dummies], axis=1)
                data = df_model.drop(explanatory, axis=1)
                explanatory_with_constant = sm.add_constant(
                    df_model.drop([response, explanatory], axis=1)
                )
            else:
                st.write(f"{explanatory} is of other data type.")
                explanatory_with_constant = sm.add_constant(df_model[explanatory])

            # without null records
            st.subheader("Simple GLM")

            model = sm.GLM(
                df_model[response],
                explanatory_with_constant,
                family=sm.families.Gaussian(),
            ).fit()

            st.write(model.summary())
            fig, ax = plt.subplots()
            y_pred = model.predict(explanatory_with_constant)
            ax.scatter(df_model[explanatory], df_model[response], color="blue")
            ax.plot(df_model[explanatory], y_pred, color="red", linewidth=2)
            st.pyplot(fig)

            df_chart = pd.DataFrame({'actual': df_model[response], 'predicted': y_pred})

            # Create a scatter plot
            fig, ax = plt.subplots()
            ax.scatter(df_chart.index, df_chart['actual'], label='Actual', color='b', alpha=0.7)
            ax.scatter(df_chart.index, df_chart['predicted'], label='Predicted', color='r', alpha=0.7)

            # Configure plot appearance
            ax.set_title('Actual vs Predicted')
            ax.set_xlabel('Index')
            ax.set_ylabel('Value')
            ax.legend()

            # Display the plot in Streamlit
            st.pyplot(fig)



            # GAM MODELS
            # Fit the GAM model
            if df_model[explanatory].dtype != "object" and df_model[response].dtype != "object":
                st.subheader("Simple GAM")
                gam = LinearGAM(n_splines=10).fit(df_model[explanatory], df_model[response])
                
                # print(gam.summary())
                # st.write(gam.summary())

                y_pred = gam.predict(df_model[explanatory])
                st.write("Predictions")
                df_test = pd.DataFrame({explanatory:df_model[explanatory], response:y_pred}).drop_duplicates()
                st.write(df_test)

                fig, ax = plt.subplots()
                ax.scatter(df_model[explanatory], df_model[response], color="blue")
                df_test = df_test.sort_values(by=explanatory)
                ax.plot(df_test[explanatory], df_test[response], color="red", linewidth=2, marker='o')
            
                st.pyplot(fig)




                df_chart = pd.DataFrame({'actual': df_model[response], 'predicted': y_pred})

                # Create a scatter plot
                fig, ax = plt.subplots()
                ax.scatter(df_chart.index, df_chart['actual'], label='Actual', color='b', alpha=0.7)
                ax.scatter(df_chart.index, df_chart['predicted'], label='Predicted', color='r', alpha=0.7)

                # Configure plot appearance
                ax.set_title('Actual vs Predicted')
                ax.set_xlabel('Index')
                ax.set_ylabel('Value')
                ax.legend()

                # Display the plot in Streamlit
                st.pyplot(fig)

    with tab4:
        pass
        # if st.button("Run data profile"):

        #     pr = df_profiling.profile_report()
        #     st_profile_report(pr)
