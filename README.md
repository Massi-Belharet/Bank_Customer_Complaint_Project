# Bank Customer Complaint Analysis

## Project Overview  
This project analyzes and classifies customer complaints submitted to the Consumer Financial Protection Bureau across five financial product categories: credit reporting, debt collection, mortgages and loans, credit cards, and retail banking. Using Natural Language Processing (NLP) and machine learning techniques, the system automatically categorizes complaint narratives to help financial institutions route them to appropriate departments more efficiently.

## Objectives
- Preprocess and analyze textual complaint data using NLP techniques  
- Develop a classification model to accurately categorize complaints based on their narratives  
- Identify key terms and patterns in different complaint categories  
- Analyze sentiment patterns across different product categories  
- Create a system that can automatically classify new complaints

## Expected Outcomes
- A robust classification model with high accuracy (achieved 88.21% with Random Forest)  
- Insights into common issues in each product category through word frequency analysis  
- Understanding of sentiment distribution across different complaint types  
- A deployable application that can classify new customer complaints  
- Recommendations for improved complaint handling based on analysis findings

## Technologies Used
- **Python**: Core programming language  
- **NLTK**: For text preprocessing, tokenization, and sentiment analysis  
- **Scikit-learn**: For machine learning models and feature extraction  
- **Pandas & NumPy**: For data manipulation and numerical operations  
- **Matplotlib & Seaborn**: For data visualization  
- **WordCloud**: For visualizing frequent terms  
- **Tkinter**: For building the desktop application interface

## How to Use
1. Clone the repository  
2. Install dependencies: `pip install -r requirements.txt`  
3. Run the notebook to train the models (model files are not included in the repository due to size)  
4. Run the application: `python complaint_classifier_app.py`  
5. Enter a complaint text in the application window  
6. Click "Classify Complaint" to see the predicted category and confidence level

## Project Structure
- `bank_complaint_analysis.ipynb`: Main notebook with data analysis and model development  
- `complaint_classifier_app.py`: Desktop application for complaint classification  
- `requirements.txt`: Required Python packages
