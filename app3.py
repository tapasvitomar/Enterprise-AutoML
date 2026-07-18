import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import joblib
from reportlab.platypus import SimpleDocTemplate, Paragraph
from reportlab.lib.styles import getSampleStyleSheet

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.utils.multiclass import type_of_target

from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import (
    RandomForestClassifier,
    RandomForestRegressor,
    GradientBoostingClassifier,
    GradientBoostingRegressor
)

from sklearn.linear_model import (
    LogisticRegression,
    LinearRegression
)

from sklearn.metrics import (
    accuracy_score,
    r2_score,
    mean_squared_error,
    classification_report,
    confusion_matrix
)

# ==========================================
# PAGE CONFIGURATION
# ==========================================

st.set_page_config(
    page_title="Enterprise AutoML & Decision Intelligence Platform",
    page_icon="🤖",
    layout="wide"
)
# ==========================================
# WEEK 7 - LOGIN AUTHENTICATION
# ==========================================

if "logged_in" not in st.session_state:
    st.session_state.logged_in = False

if not st.session_state.logged_in:

    st.title("🔐 Enterprise AutoML Login")

    username = st.text_input("Username")

    password = st.text_input(
        "Password",
        type="password"
    )

    if st.button("Login"):

        if username == "admin" and password == "admin123":

            st.session_state.logged_in = True

            st.success("Login Successful")

            st.rerun()

        else:

            st.error("Invalid Username or Password")

    st.stop()

st.title("🚀 Enterprise AutoML & Decision Intelligence Platform")

st.markdown("""
Upload any CSV or Excel dataset and automatically:

✅ Clean Data

✅ Visualize Data

✅ Train Machine Learning Models

✅ Compare Models

✅ Select Best Model

✅ Download Trained Model
""")

# ==========================================
# SIDEBAR
# ==========================================

st.sidebar.header("Dataset Upload")

uploaded_file = st.sidebar.file_uploader(
    "Upload CSV or Excel File",
    type=["csv", "xlsx"]
)

# ==========================================
# DATASET
# ==========================================

if uploaded_file is not None:

    if uploaded_file.name.endswith(".csv"):
        df = pd.read_csv(uploaded_file)

    else:
        df = pd.read_excel(uploaded_file)

    st.success("Dataset Uploaded Successfully")

    # ==========================================
    # DATASET OVERVIEW
    # ==========================================

    st.header("📊 Dataset Overview")

    c1, c2, c3 = st.columns(3)

    c1.metric("Rows", df.shape[0])

    c2.metric("Columns", df.shape[1])

    c3.metric(
        "Missing Values",
        int(df.isnull().sum().sum())
    )

    st.divider()

    # ==========================================
    # PREVIEW
    # ==========================================

    st.subheader("Dataset Preview")

    st.dataframe(df.head(10))

    # ==========================================
    # COLUMN INFO
    # ==========================================

    st.subheader("Column Information")

    info = pd.DataFrame({

        "Column": df.columns,

        "Datatype": df.dtypes.astype(str),

        "Missing": df.isnull().sum()

    })

    st.dataframe(info)

    # ==========================================
    # SUMMARY
    # ==========================================

    st.subheader("Statistical Summary")

    st.dataframe(
        df.describe(include="all")
    )

    # ==========================================
    # DATA CLEANING
    # ==========================================

    st.header("🧹 Automatic Data Cleaning")

    duplicates = df.duplicated().sum()

    st.write("Duplicate Rows:", duplicates)

    if duplicates > 0:
        df = df.drop_duplicates()

    for col in df.columns:

        if df[col].dtype == "object":

            if df[col].isnull().sum() > 0:

                df[col].fillna(
                    df[col].mode()[0],
                    inplace=True
                )

        else:

            if df[col].isnull().sum() > 0:

                df[col].fillna(
                    df[col].mean(),
                    inplace=True
                )

    st.success("Data Cleaning Completed")

    st.subheader("Cleaned Dataset")

    st.dataframe(df)

    csv = df.to_csv(index=False).encode("utf-8")

    st.download_button(
        "Download Cleaned Dataset",
        csv,
        "cleaned_dataset.csv",
        "text/csv"
    )

    st.divider()
    # ==========================================
    # WEEK 2 - DATA VISUALIZATION
    # ==========================================

    st.header("📊 Data Visualization Dashboard")

    numeric_cols = df.select_dtypes(include="number").columns.tolist()

    if len(numeric_cols) > 0:

        selected_col = st.selectbox(
            "Select Numeric Column",
            numeric_cols
        )

        # Bar Chart
        st.subheader("📊 Bar Chart")
        st.bar_chart(df[selected_col])

        # Line Chart
        st.subheader("📈 Line Chart")
        st.line_chart(df[selected_col])

        # Area Chart
        st.subheader("📉 Area Chart")
        st.area_chart(df[selected_col])

        # Histogram
        st.subheader("📦 Histogram")

        fig = px.histogram(
            df,
            x=selected_col,
            nbins=20,
            title=f"Histogram of {selected_col}"
        )

        st.plotly_chart(
            fig,
            use_container_width=True
        )

        # Box Plot
        st.subheader("📦 Box Plot")

        fig = px.box(
            df,
            y=selected_col,
            title=f"Box Plot of {selected_col}"
        )

        st.plotly_chart(
            fig,
            use_container_width=True
        )

        # Correlation Heatmap

        if len(numeric_cols) > 1:

            st.subheader("🔥 Correlation Heatmap")

            corr = df[numeric_cols].corr()

            fig = px.imshow(
                corr,
                text_auto=True,
                color_continuous_scale="Viridis"
            )

            st.plotly_chart(
                fig,
                use_container_width=True
            )

            # Scatter Plot

            st.subheader("📍 Scatter Plot")

            x_axis = st.selectbox(
                "Select X Axis",
                numeric_cols,
                key="x_axis"
            )

            y_axis = st.selectbox(
                "Select Y Axis",
                numeric_cols,
                index=1,
                key="y_axis"
            )

            fig = px.scatter(
                df,
                x=x_axis,
                y=y_axis,
                title=f"{x_axis} vs {y_axis}"
            )

            st.plotly_chart(
                fig,
                use_container_width=True
            )

        # Pie Chart

        categorical_cols = df.select_dtypes(
            include="object"
        ).columns.tolist()

        if len(categorical_cols) > 0:

            st.subheader("🥧 Pie Chart")

            cat = st.selectbox(
                "Select Category",
                categorical_cols
            )

            pie = df[cat].value_counts().reset_index()

            pie.columns = [
                cat,
                "Count"
            ]

            fig = px.pie(
                pie,
                names=cat,
                values="Count",
                title=f"{cat} Distribution"
            )

            st.plotly_chart(
                fig,
                use_container_width=True
            )

    else:

        st.warning(
            "No numeric columns found."
        )

    st.divider()
    # ==========================================
    # WEEK 3 - AUTOML ENGINE
    # ==========================================

    st.header("🤖 Enterprise AutoML Engine")

    target = st.selectbox(
        "Select Target Column",
        df.columns,
        key="target_column"
    )

    if st.button("🚀 Run AutoML"):

        data = df.copy()

        # Encode categorical columns
        encoder = LabelEncoder()

        for col in data.columns:

            if data[col].dtype == "object":

                data[col] = encoder.fit_transform(
                    data[col].astype(str)
                )

        X = data.drop(columns=[target])

        y = data[target]

        # Detect Problem Type
        target_type = type_of_target(y)

        if target_type in ["binary", "multiclass"]:

            problem = "Classification"

        else:

            problem = "Regression"

        st.success(
            f"Problem Type Detected: {problem}"
        )

        # Train Test Split
        X_train, X_test, y_train, y_test = train_test_split(
            X,
            y,
            test_size=0.20,
            random_state=42
        )

        models = {}

        results = {}

        # ==========================================
        # CLASSIFICATION
        # ==========================================

        if problem == "Classification":

            models = {

                "Decision Tree":
                DecisionTreeClassifier(random_state=42),

                "Random Forest":
                RandomForestClassifier(random_state=42),

                "Gradient Boosting":
                GradientBoostingClassifier(random_state=42),

                "Logistic Regression":
                LogisticRegression(max_iter=1000)

            }

            for name, model in models.items():

                model.fit(
                    X_train,
                    y_train
                )

                prediction = model.predict(
                    X_test
                )

                accuracy = accuracy_score(
                    y_test,
                    prediction
                )

                results[name] = round(
                    accuracy * 100,
                    2
                )

        # ==========================================
        # REGRESSION
        # ==========================================

        else:

            models = {

                "Linear Regression":
                LinearRegression(),

                "Random Forest Regressor":
                RandomForestRegressor(random_state=42),

                "Gradient Boosting Regressor":
                GradientBoostingRegressor(random_state=42)

            }

            for name, model in models.items():

                model.fit(
                    X_train,
                    y_train
                )

                prediction = model.predict(
                    X_test
                )

                score = r2_score(
                    y_test,
                    prediction
                )

                results[name] = round(
                    score * 100,
                    2
                )

        # ==========================================
        # RESULTS
        # ==========================================

        st.subheader("🏆 Model Comparison")

        result_df = pd.DataFrame({

            "Model": list(results.keys()),

            "Score": list(results.values())

        })

        st.dataframe(result_df)

        best_model = max(
            results,
            key=results.get
        )

        st.success(
            f"🏆 Best Model : {best_model}"
        )

        best = models[best_model]

        prediction = best.predict(
            X_test
        )
        # ==========================================
        # WEEK 3 - PART 3B
        # MODEL EVALUATION
        # ==========================================

        if problem == "Classification":

            st.subheader("📋 Classification Report")

            report = classification_report(
                y_test,
                prediction,
                output_dict=True
            )

            report_df = pd.DataFrame(report).transpose()

            st.dataframe(report_df)

            st.subheader("🔥 Confusion Matrix")

            cm = confusion_matrix(
                y_test,
                prediction
            )

            fig, ax = plt.subplots(figsize=(6,4))

            sns.heatmap(
                cm,
                annot=True,
                fmt="d",
                cmap="Blues",
                ax=ax
            )

            st.pyplot(fig)

        else:

            st.subheader("📈 Regression Performance")

            mse = mean_squared_error(
                y_test,
                prediction
            )

            r2 = r2_score(
                y_test,
                prediction
            )

            st.metric(
                "Mean Squared Error",
                round(mse,2)
            )

            st.metric(
                "R² Score",
                round(r2,2))

        # ==========================================
        # FEATURE IMPORTANCE
        # ==========================================

        if hasattr(best, "feature_importances_"):

            st.subheader("⭐ Feature Importance")

            importance = pd.DataFrame({

                "Feature": X.columns,

                "Importance": best.feature_importances_

            })

            importance = importance.sort_values(
                by="Importance",
                ascending=False
            )

            st.dataframe(importance)

            fig = px.bar(

                importance,

                x="Feature",

                y="Importance",

                title="Feature Importance"

            )

            st.plotly_chart(
                fig,
                use_container_width=True
            )

        # ==========================================
        # DOWNLOAD MODEL
        # ==========================================

        joblib.dump(
            best,
            "best_model.pkl"
        )

        with open(
            "best_model.pkl",
            "rb"
        ) as file:

            st.download_button(

                "💾 Download Best Model",

                file,

                file_name="best_model.pkl"

            )

        # ==========================================
        # DOWNLOAD RESULTS
        # ==========================================

        csv_result = result_df.to_csv(
            index=False
        ).encode("utf-8")

        st.download_button(

            "📄 Download AutoML Results",

            csv_result,

            file_name="automl_results.csv",

            mime="text/csv"

        )
        # ==========================================
        # WEEK 4 - DECISION INTELLIGENCE DASHBOARD
        # ==========================================

        st.divider()

        st.header("🧠 AI Decision Intelligence Dashboard")

        # Best Model Information
        best_score = results[best_model]

        col1, col2 = st.columns(2)

        with col1:
            st.metric(
                "🏆 Best Model",
                best_model
            )

        with col2:
            st.metric(
                "📈 Accuracy / Score",
                f"{best_score:.2f}%"
            )

        # AI Recommendation
        st.subheader("🤖 AI Recommendation")

        if best_score >= 95:
            st.success("Excellent model. Ready for deployment.")

        elif best_score >= 85:
            st.info("Very good model. Hyperparameter tuning may improve it further.")

        elif best_score >= 70:
            st.warning("Average model. Better feature engineering is recommended.")

        else:
            st.error("Poor model performance. Improve the dataset quality.")

        # Dataset Quality
        st.subheader("📊 Dataset Quality Report")

        quality_df = pd.DataFrame({

            "Metric":[
                "Rows",
                "Columns",
                "Missing Values",
                "Duplicate Rows"
            ],

            "Value":[
                df.shape[0],
                df.shape[1],
                int(df.isnull().sum().sum()),
                int(df.duplicated().sum())
            ]

        })

        st.dataframe(quality_df)

        # Top Important Features
        if hasattr(best, "feature_importances_"):

            st.subheader("⭐ Top 5 Important Features")

            top5 = importance.head(5)

            st.dataframe(top5)

            fig = px.bar(
                top5,
                x="Feature",
                y="Importance",
                title="Top 5 Important Features"
            )

            st.plotly_chart(
                fig,
                use_container_width=True
            )

        # AI Summary
        st.subheader("📄 Enterprise AI Summary")

        summary = f"""
### Enterprise AutoML Report

- Problem Type : **{problem}**
- Best Model : **{best_model}**
- Best Score : **{best_score:.2f}%**
- Models Trained : **{len(models)}**
- Dataset Rows : **{df.shape[0]}**
- Dataset Columns : **{df.shape[1]}**

The dataset has been automatically cleaned, visualized,
trained on multiple machine learning algorithms,
and the best-performing model has been selected.
"""

        st.markdown(summary)

        st.balloons()

        st.success("🎉 Enterprise AutoML Pipeline Completed Successfully!")
        # ==========================================
        # WEEK 5 - ENTERPRISE PREDICTION MODULE
        # ==========================================

        st.divider()

        st.header("🔮 Enterprise Prediction Module")

        st.write("Enter feature values to make a prediction using the trained AI model.")

        input_data = {}

        for feature in X.columns:

            input_data[feature] = st.number_input(
                label=f"Enter {feature}",
                value=0.0,
                key=f"input_{feature}"
            )

        if st.button("🚀 Predict New Data"):

            input_df = pd.DataFrame([input_data])

            prediction = best.predict(input_df)

            st.subheader("Prediction Result")

            if problem == "Classification":

                st.success(
                    f"Predicted Class : {prediction[0]}"
                )

            else:

                st.success(
                    f"Predicted Value : {round(float(prediction[0]),2)}"
                )

            st.divider()

            st.subheader("Prediction Summary")

            summary_df = pd.DataFrame({

                "Feature": list(input_data.keys()),

                "Entered Value": list(input_data.values())

            })

            st.dataframe(summary_df)

            st.subheader("Model Used")

            st.info(best_model)

            st.subheader("Model Score")

            st.metric(
                "Score",
                f"{best_score:.2f}%"
            )

            st.success("Prediction Completed Successfully ✅")

        st.divider()

        st.header("📈 Enterprise Dashboard")

        dashboard = pd.DataFrame({

            "Information":[
                "Problem Type",
                "Best Model",
                "Best Score",
                "Total Models",
                "Rows",
                "Columns"
            ],

            "Value":[
                problem,
                best_model,
                f"{best_score:.2f}%",
                len(models),
                df.shape[0],
                df.shape[1]
            ]

        })

        st.dataframe(dashboard)

        st.success("✅ Week 5 Completed Successfully")
        # ==========================================
        # WEEK 6 - PDF REPORT GENERATION
        # ==========================================

        st.divider()

        st.header("📄 Generate Project Report")

        if st.button("📥 Generate PDF Report"):

            styles = getSampleStyleSheet()

            pdf = SimpleDocTemplate("Enterprise_Report.pdf")

            story = []

            story.append(Paragraph(
                "<b>Enterprise AutoML & Decision Intelligence Report</b>",
                styles["Title"]
            ))

            story.append(Paragraph(
                f"<b>Problem Type:</b> {problem}",
                styles["BodyText"]
            ))

            story.append(Paragraph(
                f"<b>Best Model:</b> {best_model}",
                styles["BodyText"]
            ))

            story.append(Paragraph(
                f"<b>Best Score:</b> {best_score:.2f}%",
                styles["BodyText"]
            ))

            story.append(Paragraph(
                f"<b>Rows:</b> {df.shape[0]}",
                styles["BodyText"]
            ))

            story.append(Paragraph(
                f"<b>Columns:</b> {df.shape[1]}",
                styles["BodyText"]
            ))

            pdf.build(story)

            with open(
                "Enterprise_Report.pdf",
                "rb"
            ) as file:

                st.download_button(
                    "⬇ Download PDF Report",
                    file,
                    file_name="Enterprise_Report.pdf",
                    mime="application/pdf"
                )

            st.success("PDF Report Generated Successfully ✅")
            # ==========================================
        # WEEK 8 - MODEL PERFORMANCE DASHBOARD
        # ==========================================

        st.divider()

        st.header("📊 Enterprise Performance Dashboard")

        col1, col2, col3 = st.columns(3)

        col1.metric(
            "Rows",
            df.shape[0]
        )

        col2.metric(
            "Columns",
            df.shape[1]
        )

        col3.metric(
            "Best Score",
            f"{best_score:.2f}%"
        )

        st.divider()

        st.subheader("🏆 Best Model")

        st.success(best_model)

        st.subheader("📈 Model Ranking")

        ranking = result_df.sort_values(
            by="Score",
            ascending=False
        )

        st.dataframe(ranking)

        fig = px.bar(
            ranking,
            x="Model",
            y="Score",
            title="Model Comparison",
            color="Score"
        )

        st.plotly_chart(
            fig,
            use_container_width=True
        )

        st.subheader("📋 Enterprise Summary")

        st.info(f"""
Problem Type : {problem}

Best Model : {best_model}

Best Score : {best_score:.2f}%

Models Compared : {len(models)}

Dataset Size : {df.shape[0]} Rows × {df.shape[1]} Columns
""")

        st.success("✅ Week 8 Completed Successfully")
        # ==========================================
        # WEEK 9 - AI INSIGHTS & RECOMMENDATIONS
        # ==========================================

        st.divider()

        st.header("🧠 AI Insights & Recommendations")

        missing_values = int(df.isnull().sum().sum())
        duplicate_rows = int(df.duplicated().sum())

        st.subheader("📊 Dataset Analysis")

        insight_df = pd.DataFrame({
            "Metric": [
                "Rows",
                "Columns",
                "Missing Values",
                "Duplicate Rows",
                "Numeric Columns",
                "Categorical Columns"
            ],
            "Value": [
                df.shape[0],
                df.shape[1],
                missing_values,
                duplicate_rows,
                len(df.select_dtypes(include="number").columns),
                len(df.select_dtypes(include="object").columns)
            ]
        })

        st.dataframe(insight_df)

        st.subheader("🤖 AI Recommendations")

        recommendations = []

        if missing_values == 0:
            recommendations.append("✅ No missing values detected.")
        else:
            recommendations.append(f"⚠ {missing_values} missing values found.")

        if duplicate_rows == 0:
            recommendations.append("✅ No duplicate rows detected.")
        else:
            recommendations.append(f"⚠ {duplicate_rows} duplicate rows found.")

        if best_score >= 95:
            recommendations.append("🚀 Model is ready for deployment.")
        elif best_score >= 85:
            recommendations.append("👍 Model performs well. Hyperparameter tuning may improve it.")
        else:
            recommendations.append("📈 Consider collecting more data or improving feature engineering.")

        for rec in recommendations:
            st.write(rec)

        st.subheader("📋 Project Summary")

        st.success(f"""
Project Name : Enterprise AutoML & Decision Intelligence Platform

Problem Type : {problem}

Best Model : {best_model}

Best Score : {best_score:.2f}%

Status : Completed Successfully
""")

        st.balloons()

else:

    st.info(
        "⬅️ Please upload a CSV or Excel file from the sidebar."
    )
    
