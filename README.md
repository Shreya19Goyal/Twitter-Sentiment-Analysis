# Twitter Sentiment Analysis Project

This project aims to perform sentiment analysis on tweets using Natural Language Processing (NLP) techniques. It leverages a dataset sourced from Kaggle to train a machine learning model to classify tweets into positive or negative sentiment categories.

## Overview

Sentiment analysis plays a crucial role in understanding public opinion, customer feedback, and market trends by analyzing the sentiment expressed in text data. In this project, we focus specifically on sentiment analysis of tweets from Twitter.

## Features

- Collects tweets using the Twitter API.
- Preprocesses the text data, including tokenization, stemming, and removal of stopwords.
- Utilizes the Logistic Regression model for sentiment classification.
- Evaluates the model's performance using accuracy metrics.
- Saves and loads the trained model for future predictions.

## Dataset

The dataset used in this project is sourced from Kaggle. It contains labeled tweets, which are used for training and testing the sentiment analysis model. You can find the dataset [[here](https://www.kaggle.com/datasets/kazanova/sentiment140)](link_to_kaggle_dataset). Please make sure to comply with the dataset's license terms and conditions.

## Usage

1. Install the required dependencies using `pip install -r requirements.txt`.
2. Obtain Twitter API credentials and configure them in the `config.py` file.
3. Run the provided scripts to collect tweets, preprocess the data, and train the sentiment analysis model.
4. Evaluate the model's performance on test data and analyze the sentiment of tweets.

## Files

- `collect_tweets.py`: Script for collecting tweets using the Twitter API.
- `preprocess_data.py`: Script for preprocessing the collected tweet data.
- `sentiment_analysis.ipynb`: Jupyter Notebook containing the implementation of sentiment analysis using Logistic Regression.
- `config.py`: Configuration file for storing Twitter API credentials.
- `requirements.txt`: List of Python dependencies.
- `trained_model.sav`: Saved trained model for future predictions.

## Dependencies

- Python 3.x
- Libraries: numpy, pandas, scikit-learn, nltk

## Contributors

- [Shreya](https://github.com/Shreya19Goyal)

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
