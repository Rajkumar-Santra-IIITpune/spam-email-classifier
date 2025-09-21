# Spam Mail Classifier

This is a Flask web application that classifies emails as spam or not spam using a Naive Bayes classifier trained on a dataset.

## Setup and Running the Application Locally

1. Ensure you have Python installed on your system.

2. Navigate to the project directory:

   ```
   cd "spam mail classifier"
   ```

3. (Optional but recommended) Create and activate a virtual environment:

   - On Windows:
     ```
     python -m venv venv
     venv\Scripts\activate
     ```
   - On macOS/Linux:
     ```
     python3 -m venv venv
     source venv/bin/activate
     ```

4. Install the required dependencies:

   ```
   pip install -r requirements.txt
   ```

5. Place your dataset CSV file named `spam_ham_dataset.csv` in the project directory. The CSV file should have the following columns:

   - `text`: The email text content.
   - `label`: The label for the email (1 for spam, 0 for not spam).

6. Run the Flask application:

   ```
   python app.py
   ```

7. Open your web browser and go to:

   ```
   http://127.0.0.1:5000/
   ```

8. Enter the email text in the form and click "Classify" to see if it is spam or not.

## Deploying on Vercel

To deploy this app on Vercel:

1. Ensure your project contains the following files:
   - `vercel.json` (configuration file)
   - `wsgi.py` (entry point for the app)
   - `requirements.txt` (dependencies)
   - `app.py` (Flask app)
   - `spam_ham_dataset.csv` (dataset file)

2. Push your project to a GitHub repository.

3. Connect your GitHub repository to Vercel.

4. Vercel will automatically detect the Python app and deploy it using the `vercel.json` configuration.

5. Your app will be live on a Vercel URL.

## Notes

- The app will raise an error if the dataset file `spam_ham_dataset.csv` is not found in the project directory.
- The app drops any rows with missing values in the `text` or `label` columns before training.
- The UI uses Tailwind CSS for styling.
