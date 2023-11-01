# Psychology Chatbot for Cyber Violence Cases

This is a Python-based psychology chatbot designed to assist individuals dealing with cyber violence cases. The chatbot utilizes natural language processing techniques and a deep learning model to engage in conversations, provide support, and offer relevant information to users.

## Project Files

This repository contains the following project files:

1. **botinterface.py** - This is the main Python script that handles the chatbot's functionality. It includes the chatbot's logic, which involves processing user input, predicting user intent, and generating responses based on pre-defined intents.

2. **intents.json** - A JSON file that defines the intents and responses that the chatbot can understand and provide. It's used to train the chatbot and improve its responses.

3. **training.py** - This Python script is responsible for training the machine learning model used by the chatbot. It processes the training data, creates a model, and saves it for use by the chatbot.

## How to Use

To train the chatbot's model, follow these steps:

1. Clone the repository to your local machine using the following command:

   ```bash
   git clone https://github.com/yourusername/psychology-chatbot.git
   ```

2. Install the required Python libraries and dependencies. You can do this using a virtual environment:

   ```bash
   pip install -r requirements.txt
   ```

3. Run the training script to train the chatbot's model:

   ```bash
   python training.py
   ```

4. The training script processes the data, trains a deep learning model, and saves it as `chatbotmodel.h5` for use by the chatbot.

To run the chatbot itself, refer to the instructions in the [botinterface.py section](#how-to-run-the-chatbot) of this README.

## How to Run the Chatbot

To run the chatbot, follow these steps:

1. Ensure that the training step has been completed successfully to generate the `chatbotmodel.h5` file.

2. Open a terminal and navigate to the project directory.

3. Execute the chatbot script:

   ```bash
   python botinterface.py
   ```

4. The chatbot will start running in your terminal, allowing you to interact with it. Enter messages, and it will respond with relevant information and support related to cyber violence cases.

## Contributors

- [Ayoub Talbi](https://github.com/Ayoub-Talbi1) - Project creator and maintainer
