
# **RAG Agent ChatBot: House Color Guide Validator**

This project implements a **Retrieval-Augmented Generation (RAG) agent** for comparing house color schemes to a **best practice** guide and answering design-related questions. It uses:

- **FAISS** for vector similarity search  
- **SentenceTransformers** (`all-MiniLM-L6-v2`) for embeddings  
- **DeepSeek-7B** (`deepseek-ai/deepseek-llm-7b-chat`) as the language model  
- **scikit-learn**, **matplotlib**, and **seaborn** for evaluation and visualization  

---

## **Features**

### **1. Embedding + Retrieval**
- The system **embeds** a predefined **best practice color guide** and house descriptions using `SentenceTransformer`.
- A **FAISS index** is built for fast similarity search.

### **2. House Color Scheme Comparison**
- If the query includes **“compare house”**, the system automatically:
  - Compares **House 1** and **House 2** to the **best practice** guide.
  - Highlights **matches and mismatches** for each room.
  - Generates a **summary** of whether the house follows best practices.

### **3. General AI-Powered Answers**
- If the query is **not** a direct house comparison, the system:
  - **Retrieves relevant documents** from FAISS.
  - Uses **DeepSeek-7B** to generate a **context-aware response**.

### **4. Interactive Chat Mode**
- After evaluation, users can **enter their own design queries**.
- The chatbot **retrieves relevant context** and provides **AI-generated insights**.

### **5. Real-Time Performance Evaluation**
- Users **provide feedback** on whether the AI’s answer was correct or incorrect.
- The system **tracks accuracy, precision, recall, F1-score, and ROC-AUC**.
- Results are **visualized** with:
  - **Confusion Matrix**
  - **Bar chart of performance metrics**

---

## **Screenshots**

### **Evaluation Results and Confusion Matrix**
_(Example of model evaluation metrics based on user feedback)_

![image](https://github.com/user-attachments/assets/8ec28c67-6150-4c48-b3a9-c7df2b018351)

![image](https://github.com/user-attachments/assets/77343959-7086-4a24-9440-2222a062aafe)

![image](https://github.com/user-attachments/assets/6aa89f15-e802-40a9-96aa-b6c92faaf63a)

---



## **Installation**

### **Colab / Cloud Setup**
Run the following to install dependencies:
```bash
!pip install faiss-cpu sentence-transformers scikit-learn matplotlib seaborn transformers accelerate
```

### **Local Setup**
For local installation, use:
```bash
pip install faiss-cpu sentence-transformers scikit-learn matplotlib seaborn transformers accelerate
```

---

## **Usage**
1. **Clone or download this repository**.
2. **Run the chatbot script**:
   ```bash
   python rag_agent_chatbot.py
   ```
3. **Interact with the AI**:
   - The system first evaluates performance metrics.
   - You can then type design-related queries and receive AI-generated insights.

---

## **How It Works**

### **1. Model Initialization**
- Loads **DeepSeek-7B** for design-related text generation.
- Uses **SentenceTransformers** for document embeddings.

### **2. House Color Guide & FAISS Index**
- Hardcoded documents include:
  - **Best practice color guide**
  - **Descriptions of House 1 and House 2**
- Documents are **vectorized** and stored in **FAISS** for efficient retrieval.

### **3. Answer Generation**
- If the query includes **“compare house”**, the system:
  - Compares **each room** for correctness.
  - Lists mismatches and provides a final assessment.
- Otherwise, it **retrieves** relevant documents and **uses the LLM** to generate a **context-based response**.

### **4. Performance Evaluation**
- The chatbot **tracks accuracy** based on user feedback.
- It **visualizes performance** via:
  - **Classification reports**
  - **Confusion matrices**
  - **Bar charts of evaluation metrics**

---

## **Limitations**
- **Correctness detection** is based on word matching and user feedback.
- **LLM requires GPU resources** for optimal performance.
- **Limited to house color validation and related design queries**.

---

