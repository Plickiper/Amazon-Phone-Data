#######################
# Import libraries

import streamlit as st
import pandas as pd
import altair as alt
import plotly.express as px
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

##
from sklearn.preprocessing import MinMaxScaler

## 
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.ensemble import RandomForestRegressor
from sklearn.tree import plot_tree
import re

#######################
# Page configuration
st.set_page_config(
    page_title="Amazon Phone Data Dashboard", # Replace this with your Project's Title
    page_icon="assets/logo.png", # You may replace this with a custom icon or emoji related to your project
    layout="wide",
    initial_sidebar_state="expanded")

alt.themes.enable("dark")

#######################

# Initialize page_selection in session state if not already set
if 'page_selection' not in st.session_state:
    st.session_state.page_selection = 'about'  # Default page

# Function to update page_selection
def set_page_selection(page):
    st.session_state.page_selection = page

# Sidebar
with st.sidebar:

    # Sidebar Title (Change this with your project's title)
    st.title('Amazon Phone Data Dashboard')

    # Page Button Navigation
    st.subheader("Pages")

    if st.button("About", use_container_width=True, on_click=set_page_selection, args=('about',)):
        st.session_state.page_selection = 'about'
    
    if st.button("Dataset", use_container_width=True, on_click=set_page_selection, args=('dataset',)):
        st.session_state.page_selection = 'dataset'

    if st.button("EDA", use_container_width=True, on_click=set_page_selection, args=('eda',)):
        st.session_state.page_selection = "eda"

    if st.button("Data Cleaning / Pre-processing", use_container_width=True, on_click=set_page_selection, args=('data_cleaning',)):
        st.session_state.page_selection = "data_cleaning"

    if st.button("Machine Learning", use_container_width=True, on_click=set_page_selection, args=('machine_learning',)): 
        st.session_state.page_selection = "machine_learning"

    if st.button("Prediction", use_container_width=True, on_click=set_page_selection, args=('prediction',)): 
        st.session_state.page_selection = "prediction"

    if st.button("Conclusion", use_container_width=True, on_click=set_page_selection, args=('conclusion',)):
        st.session_state.page_selection = "conclusion"

    # Project Members
    st.subheader("Members")
    st.markdown("1. KYLE CHRYSTIAN ESPINOSA\n2. RENZO ANDRE FALLORAN\n3. CARLO LUIS LEYVA\n4. KURT JOSEPH PECENIO\n5. ANGELICA YOUNG")

#######################
# Data

# Load data
phonesearch_df = pd.read_csv("data/phone search.csv")

# Plots

# `key` parameter is used to update the plot when the page is refreshed

def histogram(column, width, height, key):
    # Generate a histogram
    histogram_chart = px.histogram(phonesearch_df, x=column)

    # Adjust the height and width
    histogram_chart.update_layout(
        width=width,   # Set the width
        height=height  # Set the height
    )

    st.plotly_chart(histogram_chart, use_container_width=True, key=f"histogram_{key}")

def scatter_plot(x_column, y_column, width, height, key):
    # Generate a scatter plot
    scatter_plot = px.scatter(phonesearch_df, x=phonesearch_df[x_column], y=phonesearch_df[y_column])

    # Adjust the height and width
    scatter_plot.update_layout(
        width=width,  # Set the width
        height=height  # Set the height
    )

    st.plotly_chart(scatter_plot, use_container_width=True, key=f"scatter_plot_{key}")
def pairwise_scatter_plot(key):
    # Generate a pairwise scatter plot matrix
    scatter_matrix = px.scatter_matrix(
        phonesearch_df,
        dimensions=['product_price', 'product_star_rating', 'product_original_price'],  # Choose relevant numeric columns
        color='is_prime'  # Replace with a categorical column if applicable
    )

    # Adjust the layout
    scatter_matrix.update_layout(
        width=500,  # Set the width
        height=500  # Set the height
    )

    st.plotly_chart(scatter_matrix, use_container_width=True, key=f"pairwise_scatter_plot_{key}")

def plot_confusion_matrix(y_true, y_pred, title="Confusion Matrix", key="conf_matrix"):
    """
    Function to plot a confusion matrix using Plotly.

    Parameters:
    y_true (list or array): True labels
    y_pred (list or array): Predicted labels
    title (str): Title of the confusion matrix plot
    key (str): Unique key for Streamlit's use_container_width functionality
    """
    # Generate the confusion matrix
    conf_matrix = confusion_matrix(y_true, y_pred)

    # Create a DataFrame for easier plotting
    conf_matrix_df = pd.DataFrame(conf_matrix, 
                                  index=['Not Amazon Choice', 'Amazon Choice'], 
                                  columns=['Not Amazon Choice', 'Amazon Choice'])

    # Create a heatmap using Plotly
    fig = px.imshow(conf_matrix_df, 
                    text_auto=True, 
                    labels={'x': 'Predicted', 'y': 'Actual'}, 
                    title=title)

    # Adjust the layout
    fig.update_layout(
        width=600,  # Width of the plot
        height=500,  # Height of the plot
    )

    # Display
    st.plotly_chart(fig, use_container_width=True, key=key)


def feature_importance_plot(feature_importance_df, width, height, key):
    # Generate a bar plot for feature importances
    feature_importance_fig = px.bar(
        feature_importance_df,
        x='Importance',
        y='Feature',
        labels={'Importance': 'Importance Score', 'Feature': 'Feature'},
        orientation='h'  # Horizontal bar plot
    )

    # Adjust the height and width
    feature_importance_fig.update_layout(
        width=width,  # Set the width
        height=height  # Set the height
    )

    st.plotly_chart(feature_importance_fig, use_container_width=True, key=f"feature_importance_plot_{key}")


# -------------------------

#######################

# Pages

# About Page
if st.session_state.page_selection == "about":
    st.header("ℹ️ About")

    st.markdown(""" 

    A Streamlit web application that performs **Exploratory Data Analysis (EDA)**, **Data Preprocessing**, and **Supervised Machine Learning** to predict "Amazon Choice" products and sales volume from the Phone Search Dataset using **Logistic Regression** and **Random Forest Regressor**.

    #### Pages
    1. `Dataset` - A brief description of the Phone Search Dataset used in this dashboard.
    2. `EDA` - Exploratory Data Analysis of the Phone Search Dataset. Highlighting the distribution of features like product price, star rating, and reviews. Includes visualizations such as histograms, scatter plots, and pairwise scatter plot matrix.
    3. `Data Cleaning / Pre-processing` - Data cleaning and pre-processing steps including encoding categorical variables, handling missing values, and splitting the dataset into training and testing sets for classification and regression tasks.
    4. `Machine Learning` - Training two supervised models: Logistic Regression for classification of "Amazon Choice" products and Random Forest Regressor for sales volume prediction. Includes model evaluation, feature importance analysis, and tree plot for Random Forest.
    5. `Prediction` - A page where users can input values to predict whether a product is "Amazon Choice" or estimate the sales volume using the trained models.
    6. `Conclusion` - Summary of the insights and observations from the EDA and model training.

    """)
   

# Dataset Page
elif st.session_state.page_selection == "dataset":
    st.header("📊 Dataset")
    st.markdown("""

    The **Amazon Phone Data dataset** contains real-time information about 340 phone products available on Amazon. It includes prices, reviews, sales volume, and tags such as Best Seller and Amazon Choice. This data is useful for analyzing price trends, forecasting sales volume, and gaining insights into e-commerce.

    For each product in the Amazon Phone Data dataset, several features are measured: asin (Amazon Standard Identification Number), product_title (name of the product), product_price (current price of the product), and product_star_rating (rating given to the product based on customer reviews). This dataset is useful for analyzing product performance, pricing strategies, and consumer preferences in e-commerce. The dataset was uploaded to Kaggle by the user shreyasur965.

    **Content**
    The dataset has **340** rows containing attributes related to phone products on Amazon. The columns include information such as Price, Reviews, Sales Volume, and Tags (e.g., Best Seller, Amazon Choice).

    `Link:` https://www.kaggle.com/datasets/shreyasur965/phone-search-dataset          
                
    """)
    
 # Display the dataset
    st.subheader("Dataset displayed as a Data Frame")
    st.dataframe(phonesearch_df, use_container_width=True, hide_index=True)
    
 # Describe Statistics
    st.subheader("Descriptive Statistics")
    st.dataframe(phonesearch_df.describe(), use_container_width=True)

    st.markdown("""

    The results from `df.describe()` highlights the descriptive statistics about the dataset. The product_star_rating has an average rating of 4.09 out of 5, with a standard deviation of 0.38, indicating that most products are rated positively, close to 4 or above. The product_num_ratings column shows an average of 2,744 ratings per product, though a large standard deviation (7,331.82) suggests high variability; while some products have very few ratings, others are extremely popular with many reviews (up to 64,977). product_num_offers averages around 8 offers per product, but again, there's substantial variability (ranging from 1 to 83), suggesting certain products are offered by many vendors. The unit_count column, with only four recorded values, suggests missing data, limiting its usefulness for analysis.

    Overall, these statistics highlight a need for further data cleaning, particularly in handling missing values in unit_count and potentially examining outliers in product_num_ratings and product_num_offers for better consistency in analysis.
                
    """)




# EDA Page
elif st.session_state.page_selection == "eda":
    st.header("📈 Exploratory Data Analysis (EDA)")

    with st.expander('Legend', expanded=True):
        st.write('''
            - Data: [Phone Search Dataset](https://www.kaggle.com/datasets/shreyasur965/phone-search-dataset).
            - :orange[**Histogram**]: Distribution of normalized product prices.
            - :orange[**Scatter Plots**]: Product prices vs. Star Ratings.
            - :orange[**Pairwise Scatter Plot Matrix**]: Highlighting *overlaps* and *differences* among Numerical features.
        ''')

    st.markdown('#### Price Distribution')
    histogram("product_price", 800, 600, 1)
    
    st.markdown('#### Product prices vs. Star Ratings')
    scatter_plot('product_price', 'product_star_rating', 500, 500, 1)
    
    st.markdown('#### Pairwise Scatter Plot Matrix')
    pairwise_scatter_plot(1)




# Data Cleaning Page
elif st.session_state.page_selection == "data_cleaning":
    st.header("🧼 Data Cleaning")

    st.subheader("Initial Dataset Preview")
    st.write("Here’s a preview of the raw dataset:")
    st.write(phonesearch_df.head())
    
    # 1 Check 
    st.subheader("Checking for Missing Values")
    missing_values = phonesearch_df.isnull().sum()
    st.write(missing_values[missing_values > 0])

    # 2 Dropping Irrelevant Columns
    st.subheader("Dropping Irrelevant Columns")
    st.code("""

    phoneData_df = phoneData_df.drop(columns=['product_url', 'product_photo'])
            
    """)
    irrelevant_columns = ['product_url', 'product_photo']
    phonesearch_df = phonesearch_df.drop(columns=irrelevant_columns)
    st.write("✅ Dropped columns:", irrelevant_columns)

    # 3 Cleaning and Converting Currency Columns
    st.subheader("Cleaning and Converting Currency Columns")
    st.code("""

    phoneData_df['product_price'] = pd.to_numeric(phoneData_df['product_price'].str.replace('[\$,]', '', regex=True))
    phoneData_df['product_original_price'] = pd.to_numeric(phoneData_df['product_original_price'].str.replace('[\$,]', '', regex=True)) 
            
    """)
    phonesearch_df['product_price'] = pd.to_numeric(phonesearch_df['product_price'].str.replace('[\$,]', '', regex=True))
    phonesearch_df['product_original_price'] = pd.to_numeric(phonesearch_df['product_original_price'].str.replace('[\$,]', '', regex=True))
    st.write("✅ Converted columns `product_price` and `product_original_price` to numeric.")
    
    # 4 Filling Missing Values with Median
    st.subheader("Filling Missing Values")
    st.code("""

    phoneData_df['product_price'] = phoneData_df['product_price'].fillna(phoneData_df['product_price'].median())
    phoneData_df['product_original_price'] = phoneData_df['product_original_price'].fillna(phoneData_df['product_original_price'].median())                  
            
    """)
    phonesearch_df['product_price'] = phonesearch_df['product_price'].fillna(phonesearch_df['product_price'].median())
    phonesearch_df['product_original_price'] = phonesearch_df['product_original_price'].fillna(phonesearch_df['product_original_price'].median())
    st.write("✅ Filled missing values with the median for `product_price` and `product_original_price`.")
    
    # 5 Outlier Removal in Product Price
    st.subheader("Outlier Removal in Product Price")
    st.code("""

    Q1 = phoneData_df['product_price'].quantile(0.25)
    Q3 = phoneData_df['product_price'].quantile(0.75)
    IQR = Q3 - Q1
    phoneData_df = phoneData_df[(phoneData_df['product_price'] >= (Q1 - 1.5 * IQR)) &
                                (phoneData_df['product_price'] <= (Q3 + 1.5 * IQR))]         
    """)
    Q1 = phonesearch_df['product_price'].quantile(0.25)
    Q3 = phonesearch_df['product_price'].quantile(0.75)
    IQR = Q3 - Q1
    phonesearch_df = phonesearch_df[(phonesearch_df['product_price'] >= (Q1 - 1.5 * IQR)) & (phonesearch_df['product_price'] <= (Q3 + 1.5 * IQR))]
    st.write("✅ Removed outliers based on the IQR method.")
    
    # 6 Normalizing Columns
    st.subheader("Normalizing Columns")
    st.code("""

    scaler = MinMaxScaler()
    phoneData_df[['product_price', 'product_star_rating']] = scaler.fit_transform(
    phoneData_df[['product_price', 'product_star_rating']])   
    
    """)
    scaler = MinMaxScaler()
    phonesearch_df[['product_price', 'product_star_rating']] = scaler.fit_transform(phonesearch_df[['product_price', 'product_star_rating']])
    st.write("✅ Normalized columns `product_price` and `product_star_rating`.")
    
    # Final preview
    st.subheader("Processed Dataset Preview")
    st.write("Here’s a preview of the processed dataset:")
    st.write(phonesearch_df.head())
 
    st.header("🧼 Data Pre-processing")
    # 1 Encoding categorical variables
    st.subheader("Encoding Categorical Variables")

    # Applying Label Encoding
    encoder = LabelEncoder()
    phonesearch_df['is_best_seller_encoded'] = encoder.fit_transform(phonesearch_df['is_best_seller'].astype(str))
    phonesearch_df['is_amazon_choice_encoded'] = encoder.fit_transform(phonesearch_df['is_amazon_choice'].astype(str))

    # Display the encoded columns
    st.write("Encoded Data (After Label Encoding):")
    st.write(phonesearch_df[['is_best_seller', 'is_best_seller_encoded', 'is_amazon_choice', 'is_amazon_choice_encoded']].head())

    # 2 Select features and target variable for classification
    st.subheader("Classification Task")
    col1, col2 = st.columns(2, gap='medium')  
    with col1:       
        X_classification = phonesearch_df[['product_price', 'product_star_rating', 'product_num_ratings']]
        y_classification = phonesearch_df['is_amazon_choice_encoded']
        # Display the selected features and target variable for classification
        st.write("Selected Features for Classification:")
        st.write(X_classification.head())
    with col2:
        st.write("Target Variable for Classification:")
        st.write(y_classification.head())
    
    # 3 Split the dataset into training and testing sets for classification
    st.subheader("Classification Data Split")
    X_train_class, X_test_class, y_train_class, y_test_class = train_test_split(X_classification, y_classification, test_size=0.3, random_state=42)
    
    # Display 
    st.write("Shape of the Training Set for Classification:")
    st.write(X_train_class.shape)
    st.write("Shape of the Test Set for Classification:")
    st.write(X_test_class.shape)

    # Save to session state 
    st.session_state['X_train_class'] = X_train_class
    st.session_state['X_test_class'] = X_test_class
    st.session_state['y_train_class'] = y_train_class
    st.session_state['y_test_class'] = y_test_class 
     
    # 4 Select features and target variable for regression
    st.subheader("Regression Task")
    X_regression = phonesearch_df[['product_price', 'product_star_rating', 'product_num_ratings']]
    y_regression = phonesearch_df['sales_volume']
    
    # Display 
    st.write("Selected Features for Regression:")
    st.write(X_regression.head())
    st.write("Target Variable for Regression:")
    st.write(y_regression.head())

    # 5 Split the dataset into training and testing sets for regression
    st.subheader("Regression Data Split")
    X_train_reg, X_test_reg, y_train_reg, y_test_reg = train_test_split(X_regression, y_regression, test_size=0.3, random_state=42)
    
    # Display 
    st.write("Shape of the Training Set for Regression:")
    st.write(X_train_reg.shape)
    st.write("Shape of the Test Set for Regression:")
    st.write(X_test_reg.shape)

    # Save the training and testing datasets to session state
    st.session_state['X_train_reg'] = X_train_reg
    st.session_state['X_test_reg'] = X_test_reg
    st.session_state['y_train_reg'] = y_train_reg
    st.session_state['y_test_reg'] = y_test_reg
    
    st.write("Class distribution in training data:")
    st.write(y_train_class.value_counts())


# Machine Learning Page
elif st.session_state.page_selection == "machine_learning":
    st.header("🤖 Machine Learning")

    # Logistic Regression Description
    st.subheader("Logistic Regression")
    st.markdown("""
    **Logistic Regression** is a `statistical` method used for binary classification problems, where the goal is to predict the probability that a given input point belongs to a particular category. Unlike linear regression, which predicts continuous values, logistic regression uses the logistic function to model a binary outcome.

    The predicted probabilities are then converted into class labels (typically 0 or 1) by applying a threshold. Commonly, a threshold of 0.5 is used, where probabilities above this threshold are classified as 1 (positive class) and those below as 0 (negative class).

    `Reference:` https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html     
    """)

    # Access X and y data from session state
    X_train_class = st.session_state['X_train_class']
    X_test_class = st.session_state['X_test_class']
    y_train_class = st.session_state['y_train_class']
    y_test_class = st.session_state['y_test_class']

    # Check the distribution of the target variable
    st.write("Class distribution in training data:")
    st.write(y_train_class.value_counts())

    # Imputer for handling missing values
    st.write("Handling missing values using median imputation...")
    imputer = SimpleImputer(strategy="median")

    # Apply the imputer to X_train_class and X_test_class
    X_train_class = imputer.fit_transform(X_train_class)
    X_test_class = imputer.transform(X_test_class)

    st.code("""
    # Initialize the imputer to replace NaN values with the median of each column
    imputer = SimpleImputer(strategy="median")

    # Apply the imputer to X_train_class and X_test_class
    X_train_class = imputer.fit_transform(X_train_class)
    X_test_class = imputer.transform(X_test_class)        
    """)
    st.write("Imputation Complete!")

    # Class Weight Handling in Logistic Regression
    st.subheader("Training the Logistic Regression model")
    log_reg_model = LogisticRegression(random_state=42, max_iter=1000, class_weight='balanced')

    # Create and fit the scaler
    scaler = MinMaxScaler()
    X_train_class_scaled = scaler.fit_transform(X_train_class)
    X_test_class_scaled = scaler.transform(X_test_class)

    # Train the Logistic Regression model on scaled data
    log_reg_model.fit(X_train_class_scaled, y_train_class)

    st.code("""
    log_reg_model = LogisticRegression(random_state=42, max_iter=1000, class_weight='balanced')
    log_reg_model.fit(X_train_class_scaled, y_train_class)     
    """)

    # Save the scaler and model to session state
    st.session_state['scaler'] = scaler
    st.session_state['log_reg_model'] = log_reg_model

    # Model Evaluation
    st.subheader("Model Evaluation")
    y_pred_class = log_reg_model.predict(X_test_class_scaled)
    accuracy_class = accuracy_score(y_test_class, y_pred_class)
    st.code("""
    y_pred_class = log_reg_model.predict(X_test_class)
    accuracy_class = accuracy_score(y_test_class, y_pred_class)
    print(f'Accuracy of Logistic Regression Classifier: {accuracy_class * 100:.2f}%')
    """)
    st.write(f"Accuracy: {accuracy_class * 100:.2f}%")

    st.markdown("""
    This implements Logistic Regression for classification. It first imputes any remaining missing values in the training and test datasets using the median. The model is then fitted on the training data, predictions are made on the test set, and the accuracy of the model is calculated and printed.
    """)

    # Classification Report
    st.subheader("Classification Report")
    classification_report_text = classification_report(y_test_class, y_pred_class)

    # Display the classification report 
    st.text(f"\nClassification Report:\n{classification_report_text}")

    # Confusion Matrix Visualization
    plot_confusion_matrix(y_test_class, y_pred_class, title="Logistic Regression Confusion Matrix")


    # Cross-Validation (after model training)
    from sklearn.model_selection import cross_val_score
    
    # Perform cross-validation
    scores = cross_val_score(log_reg_model, X_train_class_scaled, y_train_class, cv=5, scoring='accuracy')
    st.write(f"Cross-validation scores: {scores}")
    st.write(f"Mean cross-validation score: {scores.mean()}")
    
    st.markdown("""
    ### Cross-Validation Explanation:
    Cross-validation helps in evaluating the model's performance by splitting the training data into multiple folds. Each fold is used as a test set while the remaining folds are used for training. 
    This process provides an estimate of how well the model will generalize to an independent dataset. In this case, `cv=5` indicates that the data is split into 5 folds. 
    The mean cross-validation score shows the average performance across all folds, which helps us understand the model’s robustness and prevent overfitting. 
    """)
    
    # Threshold Adjustment (use a custom threshold for classifying "Amazon Choice")
    threshold = 0.5  # Adjust the threshold to your needs
    amazon_choice_prob = log_reg_model.predict_proba(X_test_class_scaled)
    amazon_choice_prediction = (amazon_choice_prob[:, 1] > threshold).astype(int)
    
    st.write(f"Custom Threshold: {threshold}")
    st.write(f"Adjusted Amazon Choice Predictions (based on threshold):")
    st.write(amazon_choice_prediction)
    
    st.markdown("""
    ### Threshold Adjustment Explanation:
    The logistic regression model outputs probabilities for each class (e.g., 'Amazon Choice' or not). By default, a threshold of 0.5 is used to decide between the two classes. 
    However, adjusting the threshold can help optimize model predictions based on business needs. For example, you may want to be more conservative in classifying a product as 'Amazon Choice' to avoid false positives. 
    In this case, we’ve set the threshold to `0.5`, which balances the trade-off between predicting 'Amazon Choice' and 'Not Amazon Choice'. 
    This can be adjusted depending on the desired level of confidence or business goals.
    """)
    
    # Final evaluation with adjusted threshold
    st.write("Final Evaluation with Adjusted Threshold")
    adjusted_accuracy = accuracy_score(y_test_class, amazon_choice_prediction)
    st.write(f"Accuracy with adjusted threshold: {adjusted_accuracy * 100:.2f}%")
    
    st.markdown("""
    ### Final Evaluation with Adjusted Threshold Explanation:
    After adjusting the classification threshold, the model’s accuracy is re-evaluated to assess its performance under the new decision boundary. 
    This helps ensure that the model is not overfitting or underfitting, and it allows for better alignment with the specific requirements for classifying products as 'Amazon Choice'. 
    In this case, the adjusted accuracy is shown as a percentage, which provides insight into the model’s real-world predictive power after threshold adjustments.
    """)


    # Random Forest
    
    # Function to extract numeric values from text-based sales volume
    def extract_numeric(value):
        if isinstance(value, str):
            numbers = re.findall(r'\d+', value)
            if numbers:
                return int(numbers[0])
            else:
                return np.nan
        return value
    
    # Clean the data (apply extract_numeric and fill NaNs with the median)
    def clean_data(y_data):
        y_data = y_data.apply(extract_numeric)
        y_data = y_data.fillna(y_data.median())
        return y_data
    
    st.subheader("Random Forest")
    st.markdown("""
    **Random Forest Regressor** is a machine learning algorithm that is used to predict continuous values by *combining multiple decision trees* which is called `"Forest"` wherein each tree is trained independently on different random subset of data and features.
    
    This process begins with data **splitting** wherein the algorithm selects various random subsets of both the data points and the features to create diverse decision trees.  
    
    Each tree is then trained separately to make predictions based on its unique subset. When it's time to make a final prediction each tree in the forest gives its own result and the Random Forest algorithm averages these predictions.
    
    `Reference:` https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestRegressor.html
    """)
    
    # Access training and testing data from session state
    X_train_reg = st.session_state['X_train_reg']
    X_test_reg = st.session_state['X_test_reg']
    y_train_reg = st.session_state['y_train_reg']
    y_test_reg = st.session_state['y_test_reg']
    
    # Clean the target variables (y_train_reg and y_test_reg)
    y_train_reg = clean_data(y_train_reg)
    y_test_reg = clean_data(y_test_reg)
    
    # Train the Random Forest Regressor model
    st.subheader("Training the Random Forest Regressor model")
    
    # Create and fit the scaler
    scaler = MinMaxScaler()
    X_train_reg_scaled = scaler.fit_transform(X_train_reg)
    X_test_reg_scaled = scaler.transform(X_test_reg)
    
    # Save the scaler to session state
    st.session_state['scaler'] = scaler
    
    # Train the Random Forest Regressor model on scaled data
    rfr_model = RandomForestRegressor(random_state=42)
    rfr_model.fit(X_train_reg_scaled, y_train_reg)
    
    st.code("""
    rfr_model = RandomForestRegressor(random_state=42)
    rfr_model.fit(X_train_reg_scaled, y_train_reg)  # Train on scaled data
    """)
    
    # Save the trained model to session state
    st.session_state['rfr_model'] = rfr_model
    
    # Model Evaluation
    st.subheader("Model Evaluation")
    
    # Evaluate model accuracy using scaled data
    train_accuracy_reg = rfr_model.score(X_train_reg_scaled, y_train_reg)
    test_accuracy_reg = rfr_model.score(X_test_reg_scaled, y_test_reg)
    
    st.write(f'Train R\u00b2 Score: {train_accuracy_reg * 100:.2f}%')
    st.write(f'Test R\u00b2 Score: {test_accuracy_reg * 100:.2f}%')
    
    st.markdown("""
    The Random Forest Regressor was trained to predict sales volume. An R² score of X% indicates how well the model explains the variance in sales volume, suggesting that the features used are relevant predictors.
    """)
    
    # Feature Importance
    st.subheader("Feature Importance")
    feature_importance = pd.Series(rfr_model.feature_importances_, index=X_train_reg.columns)
    st.bar_chart(feature_importance)


    # Visualization of Random Forest Trees
    st.subheader("Random Forest Trees Visualization")
    
    # Set a maximum number of trees to visualize
    n_estimators = min(len(rfr_model.estimators_), 20)
    
    # Set grid size for visualization
    n_rows = 4
    n_cols = 5
    
    # Ensure the grid can accommodate the number of trees
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 15), dpi=80)
    axes = axes.flatten()
    
    # Loop through the estimators and plot each tree
    for i, tree in enumerate(rfr_model.estimators_[:n_estimators]):
        plot_tree(tree, feature_names=X_train_reg.columns, filled=True, ax=axes[i])
        axes[i].set_title(f"Tree {i+1}", fontsize=8)
        axes[i].axis('off')
    
    plt.tight_layout()
    st.pyplot(fig)
    
    # Single Tree Visualization
    st.subheader("Single Tree Visualization")
    
    # Plot a single tree with a reduced figure size
    single_tree = rfr_model.estimators_[0]
    plt.figure(figsize=(14, 10))
    plot_tree(single_tree, feature_names=X_train_reg.columns, filled=True, rounded=True)
    plt.title("Single Random Forest Tree")
    st.pyplot()




# Prediction Page
elif st.session_state.page_selection == "prediction":
    st.header("👀 Prediction")

    # Check if models exist in session state
    if 'log_reg_model' not in st.session_state or 'rfr_model' not in st.session_state:
        st.error("Please train the models in the Machine Learning section first!")
        st.stop()  # Stop further execution if models are not found

    # Check if scaler exists in session state
    if 'scaler' not in st.session_state:
        st.error("Scaler is not available. Please train the models first!")
        st.stop()  # Stop further execution if scaler is not available

    st.subheader("Enter Product Metrics for Prediction")

    # Input form for prediction
    with st.form(key='prediction_form'):
        product_price = st.number_input("Product Price", min_value=0.0)  # Input for Product Price
        product_star_rating = st.number_input("Product Star Rating", min_value=0.0, max_value=5.0)  # Input for Star Rating
        product_num_ratings = st.number_input("Number of Ratings", min_value=0)  # Input for Number of Ratings
        submit_button = st.form_submit_button("Predict")

    if submit_button:
        try:
            # Prepare the input DataFrame
            input_data = pd.DataFrame({
                'product_price': [product_price],
                'product_star_rating': [product_star_rating],
                'product_num_ratings': [product_num_ratings]
            })

            # Normalize the input data using the scaler from session state
            scaler = st.session_state['scaler']  # Retrieve scaler from session state
            input_normalized = scaler.transform(input_data)

            # Make predictions using the Logistic Regression model
            amazon_choice_prob = st.session_state['log_reg_model'].predict_proba(input_normalized)
            
            # Set a custom threshold (e.g., 0.4) for predicting "Amazon Choice"
            threshold = 0.4
            amazon_choice_prediction = (amazon_choice_prob[:, 1] > threshold).astype(int)

            # Predict sales volume using Random Forest Regressor (as previously)
            sales_volume_prediction = st.session_state['rfr_model'].predict(input_normalized)

            # Format and display the prediction probabilities
            prob_no = amazon_choice_prob[0][0] * 100  # Probability for class 0 (No)
            prob_yes = amazon_choice_prob[0][1] * 100  # Probability for class 1 (Yes)
            
            st.write(f"Prediction probabilities: {amazon_choice_prob}")
            st.write(f"Probability of being 'Amazon Choice': {prob_yes:.2f}%")
            st.write(f"Probability of NOT being 'Amazon Choice': {prob_no:.2f}%")

            # Display predictions
            col1, col2 = st.columns(2)
            with col1:
                st.metric(
                    label="Amazon Choice Prediction",
                    value="Yes" if amazon_choice_prediction[0] == 1 else "No"
                )
            with col2:
                st.metric(
                    label="Predicted Sales Volume",
                    value=f"{int(sales_volume_prediction[0]):,}"
                )

        except Exception as e:
            st.error(f"An error occurred during prediction: {str(e)}")
            st.error("Please make sure all input values are valid.")




# Conclusions Page
elif st.session_state.page_selection == "conclusion":
    st.header("📝 Conclusion")

    st.markdown("""
    Through exploratory data analysis and training of two classification models (`Logistic Regression` and `Random Forest Regressor`) on the **Amazon Phone Data dataset**, the key insights and observations are:
    
    1. 📊 **Dataset Characteristics**:
       - The dataset captures valuable e-commerce metrics such as product price, ratings, and sales volume.
       - Some columns like `product_original_price`, `product_availability`, `unit_price`, and `coupon_text` had significant missing values, which were either removed or imputed as necessary.
       - Feature normalization and encoding allowed us to prepare the dataset for effective modeling.
    
    2. 📝 **Feature Distributions and Separability**:
       - Visualizations, including histograms, scatter plots, and pairwise scatter plots, revealed relationships between product prices and star ratings.
       - EDA highlighted that certain features, such as `product_price` and `product_star_rating`, have a strong correlation, suggesting potential for feature engineering in future work.
    
    3. 📈 **Model Performance (Logistic Regression for Classification)**:
    
       - The **Logistic Regression** model achieved high accuracy in predicting whether a product was designated as "Amazon Choice." This simple yet effective model was a good fit for the classification task.
       - The confusion matrix and classification report further illustrated the model’s effectiveness in classifying the products accurately based on the provided features.
    
    4. 📈 **Model Performance (Random Forest Regressor for Sales Volume Prediction)**:
       - The **Random Forest Regressor** was trained to predict product sales volume, achieving a high R² score on both the training and testing data.
       - Feature importance analysis identified `product_price` and `product_star_rating` as the most influential features for predicting sales volume, highlighting how customer preferences may affect sales.
    
    **Summing up:**  
    This project successfully demonstrated effective data preparation, exploratory analysis, and modeling techniques on the Amazon Phone Data dataset. Both models achieved high accuracy in their respective tasks, providing insights into product sales prediction and classification.
                
    """)
