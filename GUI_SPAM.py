import tkinter as tk
from tkinter import messagebox
from tkinter import scrolledtext
import pickle
import re
import nltk
from nltk.stem import WordNetLemmatizer

# Download NLTK resources if not already downloaded
nltk.download('wordnet')
nltk.download('stopwords')

# Load NLTK stopwords and lemmatizer
stopwords = nltk.corpus.stopwords.words('english')
lemmatizer = WordNetLemmatizer()

# Load CountVectorizer and Classifier from pickle files
with open('count_vectorizer.pkl', 'rb') as f:
    cv = pickle.load(f)

with open('classifier.pkl', 'rb') as f:
    classifier = pickle.load(f)

# Function to preprocess text
def preprocess_text(text):
    review = re.sub('[^a-zA-Z]', ' ', text)
    review = review.lower()
    review = review.split()
    review = [lemmatizer.lemmatize(word) for word in review if not word in set(stopwords)]
    return ' '.join(review)

# Function to perform classification
def classify_email():
    input_text = text_entry.get("1.0",'end-1c')
    if input_text.strip() == "":
        messagebox.showwarning("Warning", "Please enter an email text.")
        return
    
    processed_text = preprocess_text(input_text)
    test_vec = cv.transform([processed_text])
    prediction = classifier.predict(test_vec)[0]
    
    if prediction == 1:
        result_label.config(text="Spam Mail", fg="red")
    else:
        result_label.config(text="Not Spam Mail", fg="green")

# GUI setup
root = tk.Tk()
root.title("Spam Classifier")

# Configure grid layout
root.grid_columnconfigure(0, weight=1)
root.grid_rowconfigure(0, weight=1)

# Frame for content
frame = tk.Frame(root, padx=10, pady=10)
frame.grid(sticky="nsew")

# Text Entry Label
text_label = tk.Label(frame, text="Enter Email Text:")
text_label.grid(row=0, column=0, sticky="w")

# Expandable Text Entry with Scrollbar
text_entry = scrolledtext.ScrolledText(frame, wrap=tk.WORD, height=15, width=70)
text_entry.grid(row=1, column=0, padx=10, pady=10, sticky="nsew")

# Predict Button
predict_button = tk.Button(frame, text="Predict", command=classify_email)
predict_button.grid(row=2, column=0, pady=10, sticky="ew")

# Result Label
result_label = tk.Label(frame, text="", fg="black", font=("Helvetica", 16))
result_label.grid(row=3, column=0, pady=10)

# Configure frame resizing
frame.grid_rowconfigure(1, weight=1)
frame.grid_columnconfigure(0, weight=1)

root.mainloop()
