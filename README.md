🧠 **Sentiment Analysis Project**

This project performs Sentiment Analysis on text data collected from Reddit and Twitter. It uses machine learning and natural language processing (NLP) techniques to classify text into positive, negative, or neutral sentiments.


🚀 **Features**

Data preprocessing (cleaning, tokenization, stopword removal)

Sentiment labeling using NLP techniques

Model training using scikit-learn and pipeline serialization (joblib)

Separate sentiment pipelines for Reddit and Twitter data

Visualization of sentiment distributions and model performance

🧰 **Technologies Used**

Python 3.10+

Jupyter Notebook

scikit-learn

pandas

numpy

matplotlib / seaborn

nltk / spaCy

joblib

⚙️ **Installation**

Clone the repository:

git clone https://github.com/yourusername/Sentimental_Analysis.git
cd Sentimental_Analysis


Create and activate a virtual environment:

python -m venv myenv
myenv\Scripts\activate       # On Windows
source myenv/bin/activate    # On Mac/Linux


Install dependencies:

pip install -r requirements.txt


Launch Jupyter Notebook:

jupyter notebook

📊 **Usage**

Open the notebook sentiment_analysis2.ipynb

Run the preprocessing and model training cells

Use saved .joblib pipelines for prediction:

from joblib import load
model = load('twitter_sentiment_pipeline.joblib')
print(model.predict(["I'm feeling great today!"]))

📈 **Results**

Achieved high accuracy for sentiment classification on both datasets

Visualized positive, negative, and neutral sentiment trends across Reddit and Twitter data
