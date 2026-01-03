# Student Feedback Sentiment Analysis

## Project Overview
This project analyzes student feedback and classifies comments as positive or negative using Natural Language Processing (NLP) techniques. It uses TF-IDF for feature extraction and Logistic Regression as the classification model.

## Dataset
The dataset (`data.csv`) should contain two columns:
- `text`: The feedback text
- `label`: The sentiment (1 = Positive, 0 = Negative)

A sample dataset with around 50-100 rows is sufficient for demonstration.

## Requirements
- Python 3.8+
- pandas
- scikit-learn
- matplotlib

Install the required packages using:
```bash
pip install -r requirements.txt
```

## How to Run
1. Place `data.csv` in the same folder as `train.py`.
2. Run the script:
```bash
python train.py
```
3. The script will:
   - Load and clean the dataset
   - Extract features using TF-IDF
   - Split the data into training and testing sets
   - Train a Logistic Regression model
   - Evaluate the model and print accuracy and classification report
   - Test custom sample inputs

## Example Outputs
```
Input: The explanation was very clear and useful -> Sentiment: Positive
Input: This course is terrible and confusing -> Sentiment: Negative
```

## Usage
- Analyze student feedback for courses, teachers, or workshops
- Summarize overall satisfaction
- Use as a demo project for scholarships or GitHub portfolio
