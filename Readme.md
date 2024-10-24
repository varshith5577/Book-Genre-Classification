# Book Genre Classification

## Introduction
The **Book Genre Classification** project uses machine learning techniques to classify books based on their descriptions. The project includes a web app built using Flask, where users can input a book's description, and the app will predict its genre using a pre-trained model. The model is trained on a dataset of book descriptions and genres, utilizing text processing techniques like TF-IDF vectorization.

## Table of Contents
- [Introduction](#introduction)
- [Project Structure](#project-structure)
- [Installation](#installation)
- [Usage](#usage)
- [Model Files](#model-files)
- [Dataset](#dataset)
- [Results](#results)
- [Dependencies](#dependencies)
- [Conclusion](#conclusion)

## Project Structure
The project includes the following files:

- **app.py**: The Flask web app where users input book descriptions to get genre predictions.
- **Book Genre Classification.ipynb**: A Jupyter Notebook detailing the data analysis, model training, and evaluation process.
- **Book Genre Classification.py**: A Python script for book genre classification, including model training code.
- **BooksDataSet.csv**: The dataset containing book descriptions and corresponding genres.
- **book_genre_model.pkl**: A pre-trained machine learning model for genre prediction.
- **tfidf_vectorizer.pkl**: A pre-trained TF-IDF vectorizer for transforming book description text.
- **templates/**: A folder containing HTML templates used by the Flask web app.
- **Library.jpg**: An image used in the web interface.

## Installation
1. **Clone the Repository**:
   ```bash
   git clone <https://github.com/varshith5577/Book-Genre-Classification>
   cd Book-Genre-classification

2. **Install Dependencies**:
   Run the following command to install all required Python packages
   pip install -r requirements.txt

3. **Ensure Pre-trained Model Files**:
   book_genre_model.pkl
   tfidf_vectorizer.pkl   

## Usage

**Running the Flask Web Server**: To start the web app, run the following command,
python app.py

- This will start the Flask server at http://localhost:5000/.

**Input Data**: Enter a book description into the web form, and the app will predict the genre of the book based on the pre-trained model.

## Model Files

**book_genre_model.pkl**: The pre-trained machine learning model for genre prediction.
**tfidf_vectorizer.pkl**: The pre-trained TF-IDF vectorizer for transforming book description text data.

## Dataset

The dataset used in this project is BooksDataSet.csv, which contains book descriptions and their corresponding genres. This data was used to train the machine learning model to recognize patterns in book descriptions and classify them into genres.

## Results

The machine learning model predicts the genre of a book based on its description. The model's performance was evaluated using metrics like accuracy, precision, recall, and F1-score, which were computed during the training process in the Jupyter Notebook.

## Dependencies

To run this project, you need the following dependencies:

**Python 3.11.7**  
**Flask:** For creating the web app.  
**pandas:** For handling data operations.  
**scikit-learn:** For TF-IDF vectorization and machine learning models.  
**pickle:** To load the pre-trained model.  

Install these dependencies using:

pip install -r requirements.txt


## Conclusion

This project demonstrates a simple yet effective method of classifying books based on their descriptions using machine learning. By leveraging text processing techniques like TF-IDF and a pre-trained model, the application can predict book genres with good accuracy. The Flask web app provides an easy-to-use interface for testing the model in real-time.


**This README provides a clear and concise overview of the project, including its structure, installation instructions, usage, and other important details. Let me know if you need any further customization**!

