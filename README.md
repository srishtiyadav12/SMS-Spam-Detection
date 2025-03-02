# SMS Spam Detection App

📱 **SMS Spam Detection** is a Streamlit web application that detects whether an SMS message is spam or not using a Naive Bayes classifier.

## Features

- Detects spam messages from SMS texts
- Displays model performance metrics
- Provides a user-friendly interface for spam prediction
- Visualizes the confusion matrix of the model's performance

## Getting Started

### Prerequisites

Make sure you have Python 3.x installed. You can download it from [python.org](https://www.python.org/).

### Installation

1. **Clone the repository:**

    ```sh
    git clone https://github.com/Ysrishti-04/smsspamdetection
    cd smsspamdetection
    ```

2. **Create and activate a virtual environment:**

    ```sh
    python -m venv .venv
    source .venv/bin/activate 
    ```

3. **Install the required packages:**

    ```sh
    pip install -r requirements.txt
    ```

### Running the App

To run the Streamlit app, use the following command:

```sh
streamlit run app.py
```

Open your web browser and go to `http://localhost:8501` to see the app.

## Project Structure

```
sms-spam-detection/
│
├── project/
│   └── SMSSpamCollection    # Dataset file
├── app.py                   # Main Streamlit app
├── requirements.txt         # List of dependencies
├── README.md                # Project documentation
└── LICENSE                  # MIT License
```

## Contribution Guidelines

We welcome contributions to improve this project!

### How to Contribute

1. **Fork the repository:**

    Click the "Fork" button at the top right of this page.

2. **Clone your fork:**

    ```sh
    git clone https://github.com/Ysrishti-04/smsspamdetection
    cd smsspamdetection
    ```

3. **Create a branch:**

    ```sh
    git checkout -b feature/your-feature-name
    ```

4. **Make your changes and commit them:**

    ```sh
    git add .
    git commit -m "Add feature: your-feature-name"
    ```

5. **Push to your fork:**

    ```sh
    git push origin feature/your-feature-name
    ```

6. **Submit a pull request:**

    Go to the original repository on GitHub and click "New Pull Request". Provide a clear description of your changes.


## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.




