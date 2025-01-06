# **Spam Classification using Machine Learning**

**Name:** Soham Dutta  
**Company:** CODTECH IT Solutions  
**ID:** CT08EIL  
**Domain:** Python Programming  

---

## **Objective**  
Develop a machine learning model to classify SMS messages as "spam" or "ham" using the Naive Bayes algorithm. The project leverages text preprocessing and machine learning techniques for accurate classification.

---

## **Key Features**

1. **Dataset Management:**  
   - **Dataset:** "SMS Spam Collection" from the UCI Machine Learning Repository.  
   - **Data Download and Extraction:**  
     - Automatically downloads the dataset from the provided URL.  
     - Extracts and cleans up the dataset file after use.  

2. **Preprocessing:**  
   - Maps the labels ("ham" and "spam") to binary values (`0` for "ham" and `1` for "spam").  
   - Splits the dataset into training and testing sets using an 80-20 split ratio.  

3. **Feature Extraction:**  
   - Utilizes **CountVectorizer** to convert text messages into a bag-of-words representation for model input.  

4. **Model Building and Evaluation:**  
   - Uses **Multinomial Naive Bayes (MultinomialNB)** for spam classification.  
   - Evaluation Metrics:  
     - **Accuracy:** Overall model performance.  
     - **Classification Report:** Precision, recall, F1-score, and support for each class.  
     - **Confusion Matrix:** Summarizes predictions against actual labels.

5. **Result Storage:**  
   - Saves the evaluation results (accuracy, classification report, and confusion matrix) into a text file (`model_results.txt`) for documentation and review.

---

## **Technologies Used**

- **Python Libraries:**  
  - **`numpy`**: For numerical operations.  
  - **`pandas`**: For data manipulation and preprocessing.  
  - **`scikit-learn`**: For machine learning model training, feature extraction, and evaluation.  
  - **`requests`**: For downloading the dataset programmatically.  
  - **`zipfile` and `os`**: For handling and extracting the dataset files.  

---

## **Outcome**

The project demonstrates the development of a robust spam classification system using Naive Bayes. The model achieves high accuracy and provides detailed metrics for performance evaluation, making it a valuable tool for spam detection.

---

## **Sample Results**

- **Accuracy:** Achieved over 95% in testing.  
- **Classification Report:** Detailed metrics for "spam" and "ham" classification.  
- **Confusion Matrix:** Insights into true positives, true negatives, false positives, and false negatives.

---

## **Sample Use Case**

1. **Input:** A dataset of SMS messages with labels "ham" or "spam."  
2. **Output:**  
   - A trained Naive Bayes model that classifies new SMS messages.  
   - Performance metrics saved in `model_results.txt` for review.  

---

## **Screenshot**

![spam-email-ml-output](https://github.com/user-attachments/assets/6d3a46b0-3c5f-4f88-b12a-e8bc62405da4)

