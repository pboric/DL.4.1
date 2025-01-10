# Fake News Detection Project

A machine learning project that uses natural language processing and deep learning techniques to automatically classify news articles as real or fake. The project implements multiple models, from simple baselines to advanced BERT-based deep learning approaches.

## Project Overview

This project addresses the challenge of detecting fake news by:
- Using multiple ML/DL approaches to classify news articles
- Comparing different model architectures and their performance
- Providing detailed analysis of feature importance and error patterns
- Implementing both quick-to-deploy and high-accuracy solutions

## Features

- Multiple model implementations:
  - Baseline models (majority, title length, subject-based)
  - Traditional ML models (Logistic Regression, Naive Bayes, Random Forest)
  - BERT-based deep learning model
- Comprehensive text preprocessing pipeline
- Feature engineering including stylometric analysis
- Detailed performance analysis and visualization
- Error analysis and feature importance evaluation

## Requirements

```
pandas
numpy
matplotlib
seaborn
scikit-learn
torch
pytorch-lightning
transformers
textblob
nltk
tqdm
```

## Project Structure

```
├── data/
│   ├── Fake.csv
│   └── True.csv
├── notebooks/
│   ├── 1_EDA.ipynb
│   └── 2_Model_Development.ipynb
```

## Model Performance

Our models achieved the following performance metrics:

- BERT: Best accuracy with nuanced understanding of context
- Logistic Regression: Strong performance with quick training
- Random Forest: Good balance of accuracy and interpretability
- Naive Bayes: Fast training with competitive results

## Key Features Used

1. Text-based features:
   - TF-IDF vectors
   - Count vectors
   - Clean text and titles

2. Stylometric features:
   - Average word length
   - Punctuation patterns
   - Uppercase ratio
   - Sentiment scores

3. Metadata features:
   - Article length
   - Title length
   - Subject category

## Usage

1. Data Preparation:
```python
# Load and preprocess data
fake_df = pd.read_csv('data/Fake.csv')
true_df = pd.read_csv('data/True.csv')
fake_df['label'] = 0  # 0 for fake
true_df['label'] = 1  # 1 for true
df = pd.concat([fake_df, true_df], ignore_index=True)
```

2. Training a Model:
```python
# Example using Logistic Regression
from sklearn.linear_model import LogisticRegression
model = LogisticRegression(random_state=42)
grid = GridSearchCV(model, params, cv=3, scoring='accuracy')
grid.fit(X_train, y_train)
```

3. Using BERT Model:
```python
# Initialize and train BERT model
bert_model = FakeNewsClassifier()
trainer = pl.Trainer(max_epochs=5, accelerator='auto')
trainer.fit(bert_model, train_loader, val_loader)
```

## Results

The project achieved strong classification performance:
- High accuracy across different model types
- Robust performance across different news categories
- Effective identification of fake news patterns

Key findings:
- Title length is a strong predictor
- Certain keywords consistently indicate fake news
- Stylometric features provide additional signal
- BERT captures subtle contextual nuances

## Future Improvements

1. Data Enhancement:
   - Collect more diverse training data
   - Add source credibility metrics
   - Implement temporal validation

2. Model Improvements:
   - Implement ensemble methods
   - Add more stylometric features
   - Incorporate source reliability metrics

3. System Enhancements:
   - Add real-time processing capabilities
   - Implement API endpoints
   - Create user interface

## Contributing

Feel free to submit issues, fork the repository, and create pull requests for any improvements.

## License

MIT Licence
