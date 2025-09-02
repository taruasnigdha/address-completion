# 🚀 Address Structuring and Completion Detection

This project is a web-based application built with Streamlit that intelligently parses and validates addresses. It leverages a powerful pre-trained language model to extract structured components from unstructured address strings and a custom machine learning classifier to determine if an address is complete.

## 💡 Project Motivation

The inspiration for this project came from a machine learning design question I encountered in an interview: "How would you design an ML model to check for address completeness?" I decided to turn this interesting challenge into a mini-project to explore a practical solution.

## ✨ Key Features

-   **Address Parsing:** Extracts components like street, city, state, and zip code from a single address string.
-   **Address Validation:** Predicts whether an address is complete or incomplete using a custom-trained model.
-   **Interactive UI:** A user-friendly interface built with Streamlit to easily input and analyze addresses.
-   **Pre-trained Models:** Utilizes the `shiprocket-ai/open-llama-1b-address-completion` model for address parsing.

## 🔧 How It Works

The application uses two main components:

1.  **Language Model:** The `shiprocket-ai/open-llama-1b-address-completion` model is used to parse the unstructured address input and return a structured JSON object with the address components.
2.  **Classifier:** A custom scikit-learn model (`address_classifier_model.pkl`) predicts whether the address is complete or not. This model was trained on data sourced from Kaggle. The data was cleaned and preprocessed using the `prep_data.ipynb` notebook before being used to train the classifier.

## 🛠️ Setup and Installation

1.  **Clone the repository:**
    ```bash
    git clone https://github.com/taruasnigdha/address-completion.git
    cd address-completion
    ```

2.  **Install the dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

## ▶️ Usage

1.  **Run the Streamlit application:**
    ```bash
    streamlit run app.py
    ```

2.  **Open your browser:**
    Navigate to the URL provided by Streamlit (usually `http://localhost:8501`).

3.  **Analyze an address:**
    Enter an address in the text area and click the "Analyze Address" button to see the structured components and the completeness prediction.

## 💻 Technologies Used

-   **Streamlit:** For the web application interface.
-   **PyTorch:** For running the language model.
-   **Transformers:** For loading and using the pre-trained language model.
-   **scikit-learn:** For the address completeness classifier.
-   **Pandas & NumPy:** For data manipulation.
-   **Jupyter Notebook:** For data preparation and model training experimentation.

---

_This project was created by Snigdha Tarua._
