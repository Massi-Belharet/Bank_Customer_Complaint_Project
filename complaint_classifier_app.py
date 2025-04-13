import pickle
import tkinter as tk
from tkinter import scrolledtext, ttk
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from nltk.tokenize import TreebankWordTokenizer

# Load the pre-trained model and preprocessing objects
model = pickle.load(open('saved_models/random_forest_model.pkl', 'rb'))
vectorizer = pickle.load(open('saved_models/tfidf_vectorizer.pkl', 'rb'))
label_encoder = pickle.load(open('saved_models/label_encoder.pkl', 'rb'))

# Initialize tokenizer
tokenizer = TreebankWordTokenizer()

# Text cleaning function (same as the one used for training)
def clean_text(text):
    text = str(text).lower()
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    tokens = tokenizer.tokenize(text)
    stop_words = set(stopwords.words('english'))
    tokens = [word for word in tokens if word not in stop_words]
    ps = PorterStemmer()
    tokens = [ps.stem(word) for word in tokens]
    return ' '.join(tokens)

# Function to predict complaint category
def predict_category(text):

    cleaned_text = clean_text(text)
    
    text_vector = vectorizer.transform([cleaned_text])
    
    prediction = model.predict(text_vector)[0]
    probabilities = model.predict_proba(text_vector)[0]
    confidence = probabilities[prediction]
    
    category = label_encoder.inverse_transform([prediction])[0]
    
    return category, confidence

# Create GUI application
class ComplaintClassifierApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Bank Complaint Classifier")
        self.root.geometry("800x600")
        
        self.create_widgets()
        
    def create_widgets(self):

        title_label = tk.Label(self.root, text="Bank Complaint Classifier", font=("Arial", 16, "bold"))
        title_label.pack(pady=10)
        
        instructions = tk.Label(self.root, text="Enter a customer complaint below:")
        instructions.pack(pady=5)
        
        self.complaint_text = scrolledtext.ScrolledText(self.root, wrap=tk.WORD, width=80, height=15)
        self.complaint_text.pack(padx=10, pady=10)
        
        classify_button = tk.Button(self.root, text="Classify Complaint", command=self.classify_complaint)
        classify_button.pack(pady=10)
        
        results_frame = tk.LabelFrame(self.root, text="Classification Results")
        results_frame.pack(fill="both", expand="yes", padx=10, pady=10)
        
        self.category_label = tk.Label(results_frame, text="Category: ", font=("Arial", 12))
        self.category_label.pack(pady=5)
        
        self.confidence_label = tk.Label(results_frame, text="Confidence: ", font=("Arial", 12))
        self.confidence_label.pack(pady=5)
        
        self.confidence_bar = ttk.Progressbar(results_frame, orient="horizontal", length=300, mode="determinate")
        self.confidence_bar.pack(pady=10)
        
    def classify_complaint(self):

        complaint = self.complaint_text.get("1.0", tk.END)
        
        if complaint.strip():

            category, confidence = predict_category(complaint)
            
            self.category_label.config(text=f"Category: {category}")
            self.confidence_label.config(text=f"Confidence: {confidence:.2f}")
            
            self.confidence_bar["value"] = confidence * 100
        else:

            self.category_label.config(text="Please enter a complaint text")
            self.confidence_label.config(text="")
            self.confidence_bar["value"] = 0

# Run the application
if __name__ == "__main__":

    try:
        nltk.data.find('corpora/stopwords')
    except LookupError:
        nltk.download('stopwords')
        
    # Initialize and run the app
    root = tk.Tk()
    app = ComplaintClassifierApp(root)
    root.mainloop()