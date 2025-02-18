# RAG Agent ChatBot

This project implements a **Retrieval-Augmented Generation (RAG) agent** for comparing house color schemes to a “best practice” guide and answering design-related questions. It uses:

- **FAISS** for vector similarity search  
- **SentenceTransformers** (`all-MiniLM-L6-v2`) for embeddings  
- A **DeepSeek** language model (`deepseek-ai/deepseek-coder-1.3b-base`) for text generation  
- **scikit-learn** metrics + **matplotlib** + **seaborn** for evaluation and visualization

---

## Features

1. **Embedding + Retrieval**  
   - The script embeds each text document (best practice guide, House 1, House 2) into numeric vectors using `SentenceTransformer`.  
   - A FAISS index is built for fast similarity search.

2. **Comparison Logic**  
   - If the user’s query contains the phrase “compare house,” the code automatically compares House 1 and House 2 to the best practice guide, enumerating matches and mismatches for each room.

3. **General RAG**  
   - For other queries, the code retrieves top documents from FAISS and sends them as **context** to a **DeepSeek** LLM, generating a relevant answer.

4. **Evaluation**  
   - A small set of queries + labels is included in `test_data`.  
   - The script runs classification metrics (accuracy, precision, recall, F1, ROC AUC) on a naive yes/no correctness classification.  
   - It also visualizes the confusion matrix and a bar chart of metrics.

5. **Interactive Chat**  
   - At runtime, after evaluation, you can type your own queries, and the code will show the retrieved context plus the model’s final answer.

---

## Screenshots

Below are example screenshots from running this Colab notebook:

### 1. Evaluation Results and Confusion Matrix

![Confusion Matrix and Bar Chart]![image](https://github.com/user-attachments/assets/bbbc613c-e58f-45a8-adfc-f2ff8d3ec558)


Here, you can see the confusion matrix on the left and a bar chart of performance metrics on the right.

### 2. Classification Report and Interactive Chat

![Classification Report]![image]![image](https://github.com/user-attachments/assets/051eeaf4-ae58-4959-8570-ac9338bb52e9)



The script prints precision, recall, F1-score, and more. You can then type a query like  
`Compare house 1 and house 2 with the best practice color guide and advise if they have been painted correctly`  
and receive a detailed answer.

### 3. Comparison Query Answer

![Comparison Query]![image](https://github.com/user-attachments/assets/9d76946c-1867-42a4-9e74-7a1f36fcb161) ![image](https://github.com/user-attachments/assets/6909b281-2531-4546-b1f5-7e71a69ea110)

![image](https://github.com/user-attachments/assets/dcf0a2b9-86e4-4103-b3e3-049f5150b515)


You see the retrieved context plus a final answer describing whether House 1 or House 2 is closer to the best practice colors.

---

## Installation

In a **Colab** environment, run:

```bash
!pip install faiss-cpu sentence-transformers scikit-learn matplotlib seaborn transformers accelerate

Or locally (e.g., in a virtual environment):

```bash
pip install faiss-cpu sentence-transformers scikit-learn matplotlib seaborn transformers accelerate

## Usage
Clone or download this repository.
Open rag_agent_chatbot.py in your environment (Colab, local, etc.) that has the required libraries installed.
Run rag_agent_chatbot.py:
```bash
python rag_agent_chatbot.py

Flow:
The script evaluates the agent on test_data by running classification metrics (accuracy, confusion matrix, etc.).
Then it prompts you for queries.
Type something like:

```bash
Compare house 1 and house 2 with the best practice color guide

You’ll see the retrieved context and a final answer indicating which house is closer to best practice.

## How It Works
Initialization
Loads the SentenceTransformer embedding model and the DeepSeek 1.3b LLM in half-precision.
Documents
Hardcoded: “best_practice,” “house1,” and “house2.”
FAISS Index
We embed each document and store vectors in an IndexFlatL2.
Answer Generation
If user query includes “compare house,” the code enumerates each room’s correctness for House 1 and House 2.
Otherwise, we retrieve top docs and use the LLM to generate an answer from the combined context.
Evaluation
Minimal test data in test_data.
We parse the agent’s final answers for “correct vs. incorrect” tokens, compute standard classification metrics, and visualize them.


## Limitations
Correctness Detection
Very naive: counts certain words in the model’s output.
Small Datasets
Only a few documents and test queries. Real use requires more.
LLM Size
The 1.3B model still needs GPU resources.
Domain-Specific
Example is specific to color guides. Would need reworking for other tasks.


