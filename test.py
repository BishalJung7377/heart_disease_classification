import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import missingno as msno
from scipy import stats
import io

# Set Seaborn style globally
sns.set(style="whitegrid")

# App Title
st.title("Heart Disease Analysis Dashboard")
st.markdown("""
This interactive dashboard presents exploratory data analysis, statistical testing, 
and machine learning insights on a cleaned heart disease dataset.
""")

# Load Data
@st.cache_data
def load_data():
    df = pd.read_csv("./dataset/heart_2020_cleaned.csv")
    return df

df = load_data()

# Navigation Tabs
tabs = st.tabs(["Data Overview", "Exploratory Data Analysis (EDA)", "Statistical Tests", "Machine Learning"])

# ---------------------- Data Overview Tab ----------------------
with tabs[0]:
    st.header("Data Overview")

    st.subheader("🔍 What features characterize our data sample?")
    st.markdown("""
    ### Features that Characterize Our Data Sample

    Our data sample is characterized by a variety of features that represent demographic, behavioral, and health-related information. Some of the key features include:

    - **HeartDisease**: Target variable indicating if the individual has heart disease.
    - **BMI**: Body Mass Index, indicating body fat based on height and weight.
    - **Smoking**: Whether the individual smokes (Yes/No).
    - **AlcoholDrinking**: Whether the individual consumes alcohol heavily.
    - **Stroke**: Whether the individual has had a stroke.
    - **PhysicalHealth**: Number of days physical health was not good (in the past 30 days).
    - **MentalHealth**: Number of days mental health was not good.
    - **DiffWalking**: Indicates if the individual has serious difficulty walking.
    - **Sex**: Gender of the individual.
    - **AgeCategory**: Age group the individual falls into.
    - **Race**: Race or ethnicity.
    - **Diabetic**: Whether the person is diabetic.
    - **PhysicalActivity**: Whether the individual engages in physical activity.
    - **GenHealth**: Self-reported overall health status.
    - **SleepTime**: Average hours of sleep per day.
    - **Asthma**: Whether the individual has asthma.
    - **KidneyDisease**: Whether the individual has kidney disease.
    - **SkinCancer**: Whether the individual has had skin cancer.

    These features help us understand the individual’s lifestyle, health behaviors, and pre-existing conditions, which are important indicators in predicting the likelihood of heart disease.
    """)

    st.subheader("First 5 Rows of the Dataset")
    st.dataframe(df.head())

    st.subheader("Data Info")
    info_df = pd.DataFrame({
        'Column': df.columns,
        'Non-Null Count': df.notnull().sum().values,
        'Data Type': df.dtypes.values
    })
    st.dataframe(info_df)

    st.subheader("Shape and Duplicates")
    st.write("Dataset Shape:", df.shape)
    st.write("Duplicated Rows:", df.duplicated().sum())

    if st.checkbox("Drop duplicates"):
        df.drop_duplicates(inplace=True)
        st.success("Duplicates dropped.")
        st.write("New Shape:", df.shape)

    st.subheader("Missing Values Summary")

    # Create a DataFrame from missing value counts with appropriate headers
    missing_values = df.isna().sum().reset_index()
    missing_values.columns = ['Feature', 'Missing Values']

    # Display it in a table
    st.dataframe(missing_values)

    fig, ax = plt.subplots(figsize=(10, 5))
    sns.heatmap(df.isnull(), cbar=False, cmap="viridis", ax=ax)
    ax.set_title("Missing Values Heatmap")
    st.pyplot(fig)

    st.subheader("Unique Values and Data Types")
    unique_counts_df = pd.DataFrame({
        'Column': df.columns,
        'Unique Values': [df[col].nunique() for col in df.columns],
        'Data Type': [df[col].dtype for col in df.columns]
    })
    st.dataframe(unique_counts_df)
    

# ---------------------- EDA Tab ----------------------
with tabs[1]:
    st.header("Exploratory Data Analysis (EDA)")

    st.subheader("Target Variable Distribution")
    counts = df['HeartDisease'].value_counts()
    sizes = counts.values
    labels = ['No Heart Disease', 'Heart Disease']
    colors = ['#66b3ff', '#ff6666']

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    sns.countplot(x='HeartDisease', data=df, palette=colors, ax=ax1)
    ax1.set_title("Heart Disease Distribution")
    ax1.set_xlabel("Target")
    ax1.set_ylabel("Count")

    ax2.pie(sizes, labels=[f'{label} ({count})' for label, count in zip(labels, sizes)],
            autopct='%1.1f%%', startangle=90, colors=colors)
    ax2.axis('equal')
    ax2.set_title("Pie Chart - Heart Disease")

    st.pyplot(fig)

    st.markdown("### Feature Types")
    categorical_features = df.select_dtypes(include=['object']).columns.tolist()
    numerical_features = df.select_dtypes(include=['int64', 'float64']).columns.tolist()

    feature_types_df = pd.DataFrame({
        'Categorical Features': pd.Series(categorical_features),
        'Numerical Features': pd.Series(numerical_features)
    })
    st.dataframe(feature_types_df)

    st.subheader("📊 Numerical Feature Visualization")
    selected_numerical_feature = st.selectbox("Select a numerical feature", numerical_features)

    # Histogram with KDE
    st.header("1. Histogram with KDE")
    fig1, ax1 = plt.subplots(figsize=(10, 6))
    sns.histplot(df[selected_numerical_feature], bins=30, kde=True, color='royalblue', edgecolor='black', ax=ax1)
    ax1.set_title(f'Distribution of {selected_numerical_feature}', fontsize=16)
    ax1.set_xlabel(selected_numerical_feature)
    ax1.set_ylabel("Frequency")
    ax1.grid(True)
    st.pyplot(fig1)
    st.markdown(f"""
    **Interpretation:**
    - The distribution of **{selected_numerical_feature}** appears to be **right-skewed**, indicating that most values are clustered at the lower end, with a few high-value outliers.
    - The majority of individuals fall within a common range, suggesting a **natural central tendency** around the mode.
    - There are **some outliers or extreme values**.
    - The **smooth KDE curve** shows how the density changes and highlights potential **non-normality** in the distribution.
    """)


    # Box Plot
    st.header("2. Box Plot")
    fig2, ax2 = plt.subplots(figsize=(8, 4))
    sns.boxplot(data=df, y=selected_numerical_feature, color='lightcoral', ax=ax2)
    ax2.set_title(f'Box Plot of {selected_numerical_feature}', fontsize=14)
    ax2.set_ylabel(selected_numerical_feature)
    st.pyplot(fig2)

    st.markdown(f"""
    **Interpretation:**
    - The **box** shows the middle 50% (interquartile range).
    - The **line inside** is the **median**.
    - Points outside the **whiskers** are potential **outliers**.
    - This helps us detect skewness and extreme values in **{selected_numerical_feature}**.
    """)

    # Feature vs Heart Disease
    st.header("3. Box Plot vs Heart Disease")
    fig3, ax3 = plt.subplots(figsize=(10, 5))
    sns.boxplot(x='HeartDisease', y=selected_numerical_feature, data=df, palette='pastel', ax=ax3)
    ax3.set_title(f"{selected_numerical_feature} by Heart Disease")
    st.pyplot(fig3)

    st.markdown(f"""
    **Interpretation:**
    - This plot compares **{selected_numerical_feature}** distributions for people with and without heart disease.
    - Any large difference in medians or ranges could indicate the feature's **predictive power**.
    """)

    # Correlation Heatmap
    st.header("4. Correlation Heatmap")
    fig4, ax4 = plt.subplots(figsize=(10, 6))
    corr = df[numerical_features].corr()
    sns.heatmap(corr, annot=True, cmap="coolwarm", fmt=".2f", ax=ax4)
    ax4.set_title("Correlation Between Numerical Features")
    st.pyplot(fig4)
    # ---------------------- Categorical Feature Visualization ----------------------
    st.subheader("📋 Categorical Feature Visualization")
    selected_categorical_feature = st.selectbox("Select a categorical feature", categorical_features)

    # Count Plot

    if df[selected_categorical_feature].nunique() == 2:
        st.header("1. Distribution (Bar + Pie Chart)")
        
        counts = df[selected_categorical_feature].value_counts()
        sizes = counts.values
        labels = [f"{cat}" for cat in counts.index]
        colors = ['#66b3ff', '#ff9999']

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

        # Bar Chart
        sns.countplot(x=selected_categorical_feature, data=df, palette=colors, ax=ax1)
        ax1.set_title(f"{selected_categorical_feature} Distribution")
        ax1.set_xlabel(selected_categorical_feature)
        ax1.set_ylabel("Frequency")

        for p in ax1.patches:
            height = p.get_height()
            ax1.annotate(f'{height}', 
                        (p.get_x() + p.get_width() / 2., height), 
                        ha='center', va='bottom', fontsize=10)

        # Pie Chart
        ax2.pie(sizes,
                labels=[f'{label} ({count})' for label, count in zip(labels, sizes)],
                autopct='%1.1f%%',
                startangle=90,
                colors=colors[:len(sizes)])
        ax2.set_title(f"{selected_categorical_feature} (Pie Chart)")
        ax2.axis('equal')

        st.pyplot(fig)

        st.markdown(f"""
        **Interpretation:**
        - The **bar chart** shows the count of each category in `{selected_categorical_feature}`.
        - The **pie chart** gives a proportion-based view, helpful for spotting **imbalances**.
        - Features with binary options like `Smoking`, `Stroke`, etc., benefit from this layout.
        """)
    else:
        st.header("1. Count Plot")

        # Get counts and sort them
        value_counts = df[selected_categorical_feature].value_counts().sort_values(ascending=False)
        ordered_categories = value_counts.index.tolist()

        fig5, ax5 = plt.subplots(figsize=(12, 6))
        sns.countplot(x=selected_categorical_feature,
                    data=df,
                    order=ordered_categories,
                    color='skyblue',
                    edgecolor='black',
                    ax=ax5)

        ax5.set_title(f'Count Plot of {selected_categorical_feature}', fontsize=16)
        ax5.set_xlabel(selected_categorical_feature, fontsize=14)
        ax5.set_ylabel("Count", fontsize=14)
        plt.xticks(rotation=45, fontsize=12)
        plt.yticks(fontsize=12)

        # Add count labels on top of bars
        for bar in ax5.patches:
            height = bar.get_height()
            if height > 0:
                ax5.annotate(f'{height}', 
                            (bar.get_x() + bar.get_width() / 2., height), 
                            ha='center', va='bottom', fontsize=11, fontweight='bold', color='black')

        st.pyplot(fig5)

        st.markdown(f"""
        **Interpretation:**
        - This count plot shows how data is distributed across multiple categories in `{selected_categorical_feature}`.
        - Useful for identifying **dominant classes**, category imbalance, or rare classes.
        """)


    # Count Plot grouped by HeartDisease
    if 'HeartDisease' in df.columns:
        st.header("2. Distribution by Heart Disease")

        # Calculate total counts for percentage labeling
        total_counts = df[selected_categorical_feature].value_counts()

        fig6, ax6 = plt.subplots(figsize=(12, 6))
        palette_colors = {'Yes': '#ff9999', 'No': '#66b3ff'}

        # Create grouped count plot
        sns.countplot(x=selected_categorical_feature, hue='HeartDisease', data=df,
                    palette=palette_colors, edgecolor='black', ax=ax6)

        ax6.set_title(f'{selected_categorical_feature} by Heart Disease', fontsize=16)
        ax6.set_xlabel(selected_categorical_feature, fontsize=14)
        ax6.set_ylabel("Count", fontsize=14)
        plt.xticks(rotation=45, fontsize=12)
        plt.yticks(fontsize=12)

        # Add percentage labels on top of bars
        for container in ax6.containers:
            for bar in container:
                height = bar.get_height()
                if height > 0:
                    ax6.annotate(f'{height}', 
                                (bar.get_x() + bar.get_width() / 2., height), 
                                ha='center', va='bottom', fontsize=11, fontweight='bold', color='black')

        st.pyplot(fig6)
        st.markdown(f"""
        **Interpretation:**
        - This plot shows the breakdown of **{selected_categorical_feature}** by Heart Disease status.
        - Useful to see if certain categories have **higher or lower prevalence** of heart disease.
        - Great for identifying **categorical features with predictive value**.
        """)
        
        # Optional Outlier Removal
    st.subheader("Remove Outliers (IQR method for selected numerical features)")

    def remove_outliers_from_dataframe(df):
        for column in ['BMI', 'PhysicalHealth', 'MentalHealth', 'SleepTime']:
            if column in df.columns:
                Q1 = df[column].quantile(0.25)
                Q3 = df[column].quantile(0.75)
                IQR = Q3 - Q1
                lower_bound = Q1 - 1.5 * IQR
                upper_bound = Q3 + 1.5 * IQR
                df = df[(df[column] >= lower_bound) & (df[column] <= upper_bound)]
        return df

    if st.checkbox("Remove outliers from BMI, PhysicalHealth, MentalHealth, SleepTime"):
        df = remove_outliers_from_dataframe(df)
        st.success("Outliers removed using the IQR method.")
        st.write("New dataset shape after outlier removal:", df.shape)
        
    ### Correlation plot
    
    st.header("📈 Feature Correlation with Heart Disease")

    # Step 1: Encode and prepare data
    df_temp = df.copy()

    # Convert 'Yes'/'No' to 1/0 for correlation (temporary use only)
    df_temp.replace({'Yes': 1, 'No': 0}, inplace=True)

    # Convert categorical variables to dummy variables
    df_encoded = pd.get_dummies(df_temp, drop_first=True)

    # Ensure numeric consistency
    df_encoded = df_encoded.apply(pd.to_numeric, errors='coerce')
    df_encoded.dropna(inplace=True)

    # Step 2: Correlation Matrix
    correlation = df_encoded.corr()

    # Step 3: Select top-k features correlated with HeartDisease
    k = 18
    if 'HeartDisease' in correlation.columns:
        top_k_features = correlation['HeartDisease'].abs().nlargest(k).index
        st.write(f"Top {k} features most correlated with Heart Disease:")
        st.write(top_k_features.tolist())
    else:
        st.error("HeartDisease column not found in the correlation matrix.")
        st.stop()

    # Step 4: Generate correlation matrix for selected features
    cm = df_encoded[top_k_features].corr()

    # Step 5: Upper triangle mask
    mask = np.triu(np.ones_like(cm, dtype=bool))

    # Step 6: Plot heatmap
    fig, ax = plt.subplots(figsize=(14, 12))
    sns.heatmap(cm, mask=mask, vmax=0.8, square=True, annot=True, cmap='plasma',
                linewidths=0.5, linecolor='white', xticklabels=top_k_features, yticklabels=top_k_features,
                annot_kws={'size':12}, ax=ax)
    ax.set_title("Top Features Correlated with Heart Disease", fontsize=16, fontweight='bold')

    # Step 7: Show plot in Streamlit
    st.pyplot(fig)

    st.markdown("""
    **Interpretation:**
    - This heatmap shows the top features most correlated with Heart Disease.
    - Darker or brighter values (closer to ±1) suggest stronger relationships.
    - This is helpful for feature selection in predictive modeling.
    """)
    
    # ---------------------- Additional Visualization Section ----------------------
    st.subheader("📌 Additional Visualizations")
    if 'BMI_Category' not in df.columns:
            bins = [0, 18.5, 24.9, 29.9, 34.9, 39.9, float('inf')]
            labels = ['Underweight', 'Normal weight', 'Overweight', 'Obesity I', 'Obesity II', 'Obesity III']
            df['BMI_Category'] = pd.cut(df['BMI'], bins=bins, labels=labels, right=False)

    additional_plot_option = st.selectbox(
        "Choose an additional plot to display:",
        ["None", "BMI Categories among Heart Disease Patients"]
    )

    if additional_plot_option == "BMI Categories among Heart Disease Patients":
        # Create BMI_Category column if not already created
    
        st.markdown("### 🏥 BMI Categories Among Heart Disease Patients")
        df_heart_disease = df[df['HeartDisease'] == 'Yes']

        fig_bmi_hd, ax_bmi_hd = plt.subplots(figsize=(10, 6))
        sns.countplot(x='BMI_Category', data=df_heart_disease, palette='Spectral', ax=ax_bmi_hd)

        for p in ax_bmi_hd.patches:
            ax_bmi_hd.annotate(f'{int(p.get_height())}',
                            (p.get_x() + p.get_width() / 2., p.get_height()),
                            ha='center', va='center',
                            xytext=(0, 9),
                            textcoords='offset points')

        ax_bmi_hd.set_title('Distribution of BMI Categories for Patients with Heart Disease')
        ax_bmi_hd.set_xlabel('BMI Category')
        ax_bmi_hd.set_ylabel('Count')

        st.pyplot(fig_bmi_hd)

        st.markdown("""
        **Interpretation:**
        - This chart shows how BMI is distributed **among individuals with heart disease**.
        - Higher obesity categories could indicate greater prevalence.
        - Helpful for feature engineering and understanding health risk patterns.
        """)

# ---------------------- Statistical Tests Tab ----------------------
with tabs[2]:
    st.subheader("Select and Run Statistical Tests")

    test_choice = st.selectbox("Choose a Statistical Test:", [
        "Kolmogorov–Smirnov (K-S) Test for Normality",
        "Mann-Whitney U Test (Numerical vs Target)",
        "Chi-Square Test (Categorical vs Target)"
    ])

    df_copy = df.copy()

    if test_choice == "Kolmogorov–Smirnov (K-S) Test for Normality":
        st.markdown("""
        ### 📊 Kolmogorov–Smirnov (K-S) Test for Normality

        ---
        **Purpose:**  
        The Kolmogorov–Smirnov (K-S) test helps us assess whether the numerical features in our dataset follow a normal distribution.  
        Understanding whether features are normally distributed is crucial because many statistical tests assume normality of data.  
        If features do not follow a normal distribution, it may influence the choice of statistical tests or models.

        ---
        **Hypotheses**  
        - **Null Hypothesis (H₀):** The feature follows a normal distribution.  
        - **Alternative Hypothesis (H₁):** The feature does **not** follow a normal distribution.

        ---
        **Interpretation**  
        - If **p-value < 0.05** → Reject H₀ → Feature does **not** follow a normal distribution.  
        - If **p-value ≥ 0.05** → Fail to reject H₀ → Feature **may** follow a normal distribution.
        """)

        from sklearn.preprocessing import OrdinalEncoder
        from scipy.stats import ks_1samp, norm

        cat_cols = df_copy.select_dtypes(include='object').columns.tolist()
        df_copy[cat_cols] = OrdinalEncoder().fit_transform(df_copy[cat_cols])
        numeric_cols = df_copy.select_dtypes(include=['float64', 'int64']).columns.tolist()

        ks_results = []
        for col in numeric_cols:
            data = df_copy[col].dropna()
            standardized = (data - data.mean()) / data.std()
            stat, p = ks_1samp(standardized, norm.cdf)
            ks_results.append({
                'Feature': col,
                'KS Statistic': stat,
                'P-Value': p,
                'Conclusion': 'Not Normal (Reject H₀)' if p < 0.05 else 'Normal (Fail to Reject H₀)'
            })

        st.dataframe(pd.DataFrame(ks_results).style.format({'KS Statistic': '{:.4f}', 'P-Value': '{:.4f}'}))

    elif test_choice == "Mann-Whitney U Test (Numerical vs Target)":
        st.markdown("""
        ### 🔀 Mann-Whitney U Test (Numerical vs Target)

        ---
        **Purpose:**  
        The Mann-Whitney U test is used to compare the distributions of numerical features between two groups (e.g., people with and without heart disease).  
        Unlike the t-test, this test does not assume that the data is normally distributed, making it suitable for situations where the normality assumption is violated.  
        It helps identify features where the distributions differ, which could suggest a relationship with the target variable.

        ---
        **Hypotheses**  
        - **Null Hypothesis (H₀):** The distributions of the two groups are the same.  
        - **Alternative Hypothesis (H₁):** The distributions of the two groups are different.

        ---
        **Interpretation**  
        - If **p-value < 0.05** → Reject H₀ → Significant difference in distributions between groups.  
        - If **p-value ≥ 0.05** → Fail to reject H₀ → No significant difference in distributions.
        """)

        from scipy.stats import mannwhitneyu

        num_cols = df.select_dtypes(include=['float64', 'int64']).columns.tolist()
        results = []
        for col in num_cols:
            g1 = df[df['HeartDisease'] == 'Yes'][col]
            g2 = df[df['HeartDisease'] == 'No'][col]
            stat, p = mannwhitneyu(g1, g2, alternative='two-sided')
            results.append({
                'Feature': col,
                'U-Statistic': stat,
                'P-Value': p,
                'Conclusion': 'Different Distributions (Reject H₀)' if p < 0.05 else 'Same Distribution (Fail to Reject H₀)'
            })

        st.dataframe(pd.DataFrame(results).style.format({'U-Statistic': '{:.2f}', 'P-Value': '{:.4f}'}))

    elif test_choice == "Chi-Square Test (Categorical vs Target)":
        st.markdown("""
        ### 🔢 Chi-Square Test of Independence (Categorical vs Target)

        ---
        **Purpose:**  
        The Chi-Square test helps us determine if there is a statistical association between categorical features and the target variable (HeartDisease).  
        This test is useful for identifying categorical features that might be predictive of heart disease, such as smoking habits, stroke history, or diabetes.  
        A significant association suggests that the feature plays a role in differentiating between the heart disease groups.

        ---
        **Hypotheses**  
        - **Null Hypothesis (H₀):** There is no association between the categorical feature and heart disease.  
        - **Alternative Hypothesis (H₁):** There is an association between the categorical feature and heart disease.

        ---
        **Interpretation**  
        - If **p-value < 0.05** → Reject H₀ → There is an association between the feature and heart disease.  
        - If **p-value ≥ 0.05** → Fail to reject H₀ → No significant association between the feature and heart disease.
        """)

        from scipy.stats import chi2_contingency

        cat_cols = df.select_dtypes(include='object').columns.tolist()
        if 'HeartDisease' in cat_cols:
            cat_cols.remove('HeartDisease')

        results = []
        for col in cat_cols:
            table = pd.crosstab(df[col], df['HeartDisease'])
            chi2, p, dof, expected = chi2_contingency(table)
            results.append({
                'Feature': col,
                'Chi-Square Statistic': chi2,
                'P-Value': p,
                'Degrees of Freedom': dof,
                'Conclusion': 'Associated (Reject H₀)' if p < 0.05 else 'Independent (Fail to Reject H₀)'
            })

        st.dataframe(pd.DataFrame(results).style.format({'Chi-Square Statistic': '{:.2f}', 'P-Value': '{:.4f}'}))


# ---------------------- Machine Learning Tab ----------------------
with tabs[3]:
    st.header("Machine Learning")

    from sklearn.model_selection import train_test_split, GridSearchCV
    from sklearn.linear_model import LogisticRegression
    from sklearn.svm import SVC
    from sklearn.neural_network import MLPClassifier
    from sklearn.neighbors import KNeighborsClassifier
    from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
    from sklearn.preprocessing import OrdinalEncoder, OneHotEncoder, StandardScaler, LabelEncoder
    from sklearn.compose import ColumnTransformer
    from sklearn.decomposition import PCA
    from imblearn.over_sampling import SMOTE
    from collections import Counter
    from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

    # --- Data Preprocessing ---
    all_ordinal_cols = ['BMI_Category', 'AgeCategory', 'Race', 'GenHealth']
    all_boolean_cols = [col for col in df.columns if df[col].nunique() == 2]

    ordinal_cols = [col for col in all_ordinal_cols if col in df.columns]
    boolean_cols = [col for col in all_boolean_cols if col in df.columns]

    ordinal_mappings = {
        'BMI_Category': ['Underweight', 'Normal weight', 'Overweight', 'Obesity I', 'Obesity II', 'Obesity III'],
        'AgeCategory': ['18-24', '25-29', '30-34', '35-39', '40-44', '45-49', '50-54', '55-59', '60-64', '65-69', '70-74', '75-79', '80 or older'],
        'Race': ['White', 'Black', 'Asian','Hispanic', 'American Indian/Alaskan Native', 'Other'],
        'GenHealth': ['Poor', 'Fair', 'Good', 'Very good', 'Excellent']
    }


    preprocessor = ColumnTransformer(
        transformers=[
            ('ord', OrdinalEncoder(categories=[ordinal_mappings[col] for col in ordinal_cols]), ordinal_cols),  # Ordinal encoding
            ('ohe', OneHotEncoder(drop='first'), boolean_cols)  # OneHotEncoding for boolean columns
        ],
        remainder='passthrough'  
    )


    df_transformed = preprocessor.fit_transform(df)

    # Convert the transformed data back to a DataFrame with appropriate column names
    # Ordinal columns retain original names, while OneHotEncoder generates new columns
    ohe_columns = preprocessor.named_transformers_['ohe'].get_feature_names_out(boolean_cols)
    final_columns = ordinal_cols + list(ohe_columns) + [col for col in df.columns if col not in ordinal_cols + boolean_cols]

    df_encoded = pd.DataFrame(df_transformed, columns=final_columns)

    st.subheader("Preview of Encoded Dataset")
    st.dataframe(df_encoded.head())

    st.subheader("Encoded Data Information")
    info_df = pd.DataFrame({
        'Column': df_encoded.columns,
        'Data Type': df_encoded.dtypes.values,
        'Non-Null Count': df_encoded.notnull().sum().values
    })
    st.dataframe(info_df)

    st.header("PCA & Sampling")

    use_full_data = st.checkbox("Use full dataset for training (slower)", value=False)
    df_active = df_encoded if use_full_data else df_encoded.sample(frac=0.3, random_state=42)

    X = df_active.drop('HeartDisease_Yes', axis=1)
    y = df_active['HeartDisease_Yes'].astype(int)

    X_encoded = X.copy()
    label_encoders = {}
    for col in X_encoded.select_dtypes(include=['object']).columns:
        le = LabelEncoder()
        X_encoded[col] = le.fit_transform(X_encoded[col])
        label_encoders[col] = le

    scaler = StandardScaler()
    features_scaled = scaler.fit_transform(X_encoded)
    st.dataframe(X_encoded)
    pca = PCA()
    pca_result = pca.fit_transform(features_scaled)
    cumulative_variance = np.cumsum(pca.explained_variance_ratio_)
    n_components = np.argmax(cumulative_variance >= 0.95) + 1
    X_scaled = pca_result[:, :n_components]
    
    st.success(f"PCA reduced the feature space to {n_components} components (95%+ variance retained).")

    if st.checkbox("Show PCA Variance Plot"):
            explained_variance_data = pd.DataFrame({
                'PC': [f'PC{i+1}' for i in range(len(cumulative_variance))],
                'Explained Variance (%)': pca.explained_variance_ratio_ * 100
            })
            fig, ax = plt.subplots(figsize=(12, 6))
            palette = sns.color_palette("coolwarm", n_colors=len(explained_variance_data[:20]))
            sns.barplot(x='PC', y='Explained Variance (%)', data=explained_variance_data[:20], palette=palette, ax=ax)
            for i, val in enumerate(explained_variance_data['Explained Variance (%)'][:20]):
                ax.text(i, val + 0.5, f"{val:.2f}%", ha='center', va='bottom', fontsize=10)
            ax.axhline(y=(1.0 / len(X.columns)) * 100, color='red', linestyle='--', label='Random Baseline')
            ax.set_title('Explained Variance by PCA Components')
            ax.legend()
            st.pyplot(fig)

    # Helper plotting functions
    def plot_confusion_heatmap(cm, dataset_name, model_name):
        fig, ax = plt.subplots()
        sns.heatmap(cm, annot=True, fmt='g', cmap='Blues', ax=ax,
                    xticklabels=['No Heart Disease', 'Heart Disease'],
                    yticklabels=['No Heart Disease', 'Heart Disease'])
        ax.set_title(f'Confusion Matrix - {model_name} ({dataset_name})')
        ax.set_xlabel("Predicted")
        ax.set_ylabel("Actual")
        st.pyplot(fig)

    def plot_classification_report_heatmap(report, dataset_name, model_name):
        # Convert classification report to DataFrame
        report_df = pd.DataFrame(report).transpose()

        # Select only class labels (exclude avg/total rows)
        class_labels = ['No Heart Disease', 'Heart Disease']
        report_df = report_df.loc[class_labels, ['precision', 'recall', 'f1-score', 'support']]

        # Rename columns for clean titles
        report_df.columns = ['Precision', 'Recall', 'F1-Score', 'Support']
        report_df.index.name = 'Class'

        # Round for display
        display_df = report_df.copy()
        display_df.iloc[:, :3] = display_df.iloc[:, :3].round(2)
        display_df['Support'] = display_df['Support'].astype(int)

        # Plot
        fig, ax = plt.subplots(figsize=(10, 5))
        sns.heatmap(display_df, annot=True, fmt='.2f', cmap='Blues', cbar=True, linewidths=0.5, linecolor='gray')

        ax.set_title(f'Classification Report Heatmap\n{model_name} on {dataset_name}', fontsize=14, weight='bold')
        ax.set_xlabel('Metrics')
        ax.set_ylabel('Class')

        st.pyplot(fig)

    # Define the training function
    def train_and_evaluate_model(model, dataset_name, model_name, 
                                sampler=None, hyperparameters=None, param_grid=None, 
                                test_size=0.33, random_state=42):

        X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=test_size, random_state=random_state)

        if sampler:
            X_train, y_train = sampler.fit_resample(X_train, y_train)

        if param_grid:
            grid_search = GridSearchCV(model, param_grid, cv=5, scoring='accuracy', n_jobs=-1)
            grid_search.fit(X_train, y_train)
            model = grid_search.best_estimator_
            st.write(f"🔍 Best parameters found for {model_name}:", grid_search.best_params_)

        elif hyperparameters:
            try:
                model.set_params(**hyperparameters)
            except ValueError as e:
                st.warning(f"⚠️ Some parameters may not be applicable for {model_name}: {e}")

        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        accuracy = accuracy_score(y_test, y_pred)
        report = report = classification_report(y_test, y_pred, output_dict=True, target_names=["No Heart Disease", "Heart Disease"])
        cm = confusion_matrix(y_test, y_pred)

        return {
            'model_name': model_name,
            'dataset_name': dataset_name,
            'accuracy': accuracy,
            'classification_report': report,
            'confusion_matrix': cm
        }

    # --- UI Part ---
    st.subheader("Select a Classification Model")

    model_option = st.selectbox(
        "Choose a model", 
        ["Logistic Regression", "Random Forest", "SVM", "Gradient Boosting", "Multilayer Perceptron (MLP)", "K-Nearest Neighbors (KNN)"]
    )

    apply_smote = st.checkbox("Apply SMOTE for balancing", value=False)
    tune_hyperparams = st.checkbox("Enable GridSearchCV (Hyperparameter Tuning)", value=False)

    # Optional hyperparameters
    param_grid_map = {
        "Logistic Regression": {
            'C': [0.01, 0.1, 1, 10],
            'penalty': ['l2'],
            'solver': ['liblinear'],
            'max_iter': [100, 300, 500]
        },
        "Random Forest": {
            'n_estimators': [50, 100, 200],
            'max_features': ['auto', 'sqrt', 'log2'],
            'max_depth' : [4,5,6,7,8],
            'criterion' :['gini', 'entropy'],
        },
        "SVM": {
            'C': [0.1, 1, 10, 100],
            'kernel': ['linear', 'rbf', 'poly'],
            'gamma': ['scale', 'auto', 0.001, 0.01, 0.1, 1],
            'degree': [2, 3, 4]
        },
        "Gradient Boosting": {
             'n_estimators': [100, 200, 300],          # More trees can improve performance but increase training time
            'learning_rate': [0.01, 0.05, 0.1],       # Lower = slower learning, but potentially better accuracy
            'max_depth': [3, 4, 5, 6],                # Controls model complexity
            'subsample': [0.6, 0.8, 1.0],             # Prevent overfitting by using a fraction of samples for training each tree
            'min_samples_split': [2, 5, 10],          # Minimum samples to split an internal node
            'min_samples_leaf': [1, 2, 4],            # Minimum samples at a leaf node
            'max_features': ['sqrt', 'log2', None]
        },
        "Multilayer Perceptron (MLP)": {
            'hidden_layer_sizes': [(5,), (10,), (7, 7)],
            'activation': ['relu', 'tanh'],
            'solver': ['sgd','adam'],  # Using stochastic gradient descent
            'learning_rate': ['constant', 'adaptive'],
            'alpha': [0.0001, 0.001, 0.01],
            'max_iter': [200, 500]
        },
        "K-Nearest Neighbors (KNN)": {
            'n_neighbors': [3, 5, 7],
            'weights': ['uniform', 'distance'],
            'metric': ['euclidean', 'manhattan']
        }
    }

    # Model map
    model_map = {
        "Logistic Regression": LogisticRegression(max_iter=1000),
        "Random Forest": RandomForestClassifier(),
        "SVM": SVC(probability=True),
        "Gradient Boosting": GradientBoostingClassifier(),
        "Multilayer Perceptron (MLP)": MLPClassifier(max_iter=500),
        "K-Nearest Neighbors (KNN)": KNeighborsClassifier()
    }

    # Select model and parameters
    selected_model = model_map[model_option]
    param_grid = param_grid_map.get(model_option) if tune_hyperparams else None
    sampler = SMOTE(random_state=42) if apply_smote else None

    if st.button("Train Model"):
        results = train_and_evaluate_model(
            model=selected_model,
            dataset_name="Heart Disease",
            model_name=model_option,
            sampler=sampler,
            param_grid=param_grid
        )

        st.subheader(f"🔍 Evaluation: {model_option}")
        st.write(f"✅ Accuracy: **{results['accuracy']:.4f}**")

        plot_confusion_heatmap(results['confusion_matrix'], "Heart Disease", model_option)
        plot_classification_report_heatmap(results['classification_report'], "Heart Disease", model_option)

        st.subheader("📄 Raw Classification Report Table")
        st.dataframe(pd.DataFrame(results['classification_report']).transpose().style.format("{:.2f}"))
        
        
        ## provide all model accuracy
    # st.header("📊 Compare Base vs Hyperparameter-Tuned Models")

    # # Multiselect for model comparison
    # selected_models = st.multiselect(
    #     "Select models to compare",
    #     options=list(model_map.keys()),
    #     default=["Logistic Regression", "Random Forest", "SVM"]
    # )

    # # Let user choose comparison mode
    # compare_tuned = st.checkbox("Also compare hyperparameter-tuned versions", value=True)

    # if st.button("Run Selected Models"):
    #     base_accuracies = []
    #     tuned_accuracies = []

    #     for model_name in selected_models:
    #         st.write(f"🔹 Training base {model_name}...")
    #         base_model = model_map[model_name]

    #         # Train base model
    #         base_result = train_and_evaluate_model(
    #             model=base_model,
    #             dataset_name="Heart Disease",
    #             model_name=model_name,
    #             sampler=sampler,
    #             param_grid=None  # No tuning
    #         )
    #         base_accuracies.append((model_name, base_result['accuracy']))

    #         if compare_tuned:
    #             st.write(f"🔸 Training hyperparameter-tuned {model_name}...")
    #             tuned_model = model_map[model_name]
    #             param_grid = param_grid_map.get(model_name)

    #             tuned_result = train_and_evaluate_model(
    #                 model=tuned_model,
    #                 dataset_name="Heart Disease",
    #                 model_name=f"{model_name} (Tuned)",
    #                 sampler=sampler,
    #                 param_grid=param_grid
    #             )
    #             tuned_accuracies.append((model_name, tuned_result['accuracy']))

    #     # Convert to DataFrame
    #     df_base = pd.DataFrame(base_accuracies, columns=["Model", "Base Accuracy"])
    #     if compare_tuned:
    #         df_tuned = pd.DataFrame(tuned_accuracies, columns=["Model", "Tuned Accuracy"])
    #         df_all = pd.merge(df_base, df_tuned, on="Model")
    #     else:
    #         df_all = df_base.copy()

    #     # Plot
    #     fig, ax = plt.subplots(figsize=(12, 6))

    #     ax.plot(df_all["Model"], df_all["Base Accuracy"], marker='o', linestyle='-', label='Base Accuracy', color='blue')
    #     if compare_tuned:
    #         ax.plot(df_all["Model"], df_all["Tuned Accuracy"], marker='s', linestyle='--', label='Tuned Accuracy', color='orange')

    #     # Label points
    #     for i, row in df_all.iterrows():
    #         ax.text(i, row['Base Accuracy'] + 0.005, f"{row['Base Accuracy']*100:.2f}%", ha='center', fontsize=9, color='blue')
    #         if compare_tuned:
    #             ax.text(i, row['Tuned Accuracy'] + 0.005, f"{row['Tuned Accuracy']*100:.2f}%", ha='center', fontsize=9, color='orange')

    #     ax.set_title("🔬 Model Accuracy Comparison: Base vs Hyperparameter-Tuned", fontsize=14, weight='bold')
    #     ax.set_ylabel("Accuracy")
    #     ax.set_ylim(0.7, 1.0)
    #     ax.grid(True, axis='y')
    #     ax.legend()
    #     plt.xticks(rotation=45, ha='right')

    #     st.pyplot(fig)

    #     # Show results as table
    #     st.subheader("📄 Accuracy Table")
    #     if compare_tuned:
    #         st.dataframe(df_all.style.format({"Base Accuracy": "{:.4f}", "Tuned Accuracy": "{:.4f}"}))
    #     else:
    #         st.dataframe(df_base.style.format({"Base Accuracy": "{:.4f}"}))
    st.header("📊 Compare Base vs Hyperparameter-Tuned Models")

    # Multiselect for model comparison
    selected_models = st.multiselect(
        "Select models to compare",
        options=list(model_map.keys()),
        default=["Logistic Regression", "Random Forest", "SVM"]
    )

    # User chooses whether to run tuned version
    compare_tuned = st.checkbox("Also compare hyperparameter-tuned versions", value=True)

    if st.button("Run Selected Models"):
        base_accuracies = []
        tuned_accuracies = []

        for model_name in selected_models:
            st.write(f"🔹 Training base {model_name}...")
            base_model = model_map[model_name]

            # Train base model
            base_result = train_and_evaluate_model(
                model=base_model,
                dataset_name="Heart Disease",
                model_name=model_name,
                sampler=sampler,
                param_grid=None
            )
            base_accuracies.append((model_name, base_result['accuracy']))

            # Train tuned model if selected
            if compare_tuned:
                st.write(f"🔸 Training hyperparameter-tuned {model_name}...")
                tuned_model = model_map[model_name]
                param_grid = param_grid_map.get(model_name)

                tuned_result = train_and_evaluate_model(
                    model=tuned_model,
                    dataset_name="Heart Disease",
                    model_name=f"{model_name} (Tuned)",
                    sampler=sampler,
                    param_grid=param_grid
                )
                tuned_accuracies.append((model_name, tuned_result['accuracy']))

        # Convert to DataFrames
        df_base = pd.DataFrame(base_accuracies, columns=["Model", "Base Accuracy"])
        df_all = df_base.copy()

        has_base = len(base_accuracies) > 0
        has_tuned = compare_tuned and len(tuned_accuracies) > 0

        if has_tuned:
            df_tuned = pd.DataFrame(tuned_accuracies, columns=["Model", "Tuned Accuracy"])
            df_all = pd.merge(df_base, df_tuned, on="Model", how="outer")

        # Plotting
        fig, ax = plt.subplots(figsize=(12, 6))

        # Plot base accuracy
        if has_base:
            ax.plot(df_all["Model"], df_all["Base Accuracy"], marker='o', linestyle='-', color='blue', label='Base Accuracy')
            for i, row in df_all.iterrows():
                if pd.notnull(row['Base Accuracy']):
                    ax.text(i, row['Base Accuracy'] + 0.005, f"{row['Base Accuracy']*100:.2f}%", ha='center', fontsize=9, color='blue')

        # Plot tuned accuracy
        if has_tuned:
            ax.plot(df_all["Model"], df_all["Tuned Accuracy"], marker='s', linestyle='--', color='orange', label='Tuned Accuracy')
            for i, row in df_all.iterrows():
                if pd.notnull(row['Tuned Accuracy']):
                    ax.text(i, row['Tuned Accuracy'] + 0.005, f"{row['Tuned Accuracy']*100:.2f}%", ha='center', fontsize=9, color='orange')

        # Dynamic title
        if has_base and has_tuned:
            title = "🔬 Model Accuracy Comparison: Base vs Hyperparameter-Tuned"
        elif has_base:
            title = "🟦 Model Accuracy: Base Models"
        elif has_tuned:
            title = "🟧 Model Accuracy: Hyperparameter-Tuned Models"
        else:
            title = "Model Accuracy Comparison"

        ax.set_title(title, fontsize=14, weight='bold')
        ax.set_ylabel("Accuracy")
        ax.set_ylim(0.7, 1.0)
        ax.grid(True, axis='y')
        ax.legend()
        plt.xticks(rotation=45, ha='right')

        st.pyplot(fig)

        # Show results table
        st.subheader("📄 Accuracy Table")
        display_df = df_all.copy()
        accuracy_cols = [col for col in display_df.columns if "Accuracy" in col]
        st.dataframe(display_df.style.format({col: "{:.4f}" for col in accuracy_cols}))

   
# Footer
st.markdown("---")
st.caption("Developed by Bishal Jung  using Streamlit")
