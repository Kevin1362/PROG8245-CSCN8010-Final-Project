# 🎓 Campus Support Agent

This project keeps the original **Campus Support Assistant** domain and upgrades it into a lightweight **agent**.

## What the agent adds

- Intent detection for question types such as hours, location, contact, and procedures
- Short-term memory for follow-up questions such as **"Where is it?"**
- Automatic fallback guidance when confidence is low
- Choice of **Auto**, **GloVe**, or **Word2Vec** retrieval
- Gradio chat interface with retrieval details

## Project structure

```text
PROG8245-CSCN8010-Final-Project-Dry-Run/
│
├── app.py
├── data/
│   └── knowledge_base.json
├── models/
│   └── word2vec_campus.bin
├── documentation/
│   └── architecture.md
├── notebooks/
│   └── Campus_Support_Assistant.ipynb
├── src/
│   ├── agent.py
│   ├── tools.py
│   ├── data_processing.py
│   ├── evaluate.py
│   ├── predict.py
│   └── train.py
└── requirements.txt
```

## How to run

### 1. Create and activate a virtual environment

**Windows PowerShell**

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Run the app

```bash
python app.py
```

Then open:

```text
http://localhost:7860
```

## Optional retraining

If you update the knowledge base and want to retrain the Word2Vec model:

```bash
python src/train.py
```

## Example questions

- What time does the library open?
- Where is the financial aid office?
- How can I contact student services?
- Is there a gym on campus?
- Where is it?

## Notes

- The project still uses your original campus FAQ knowledge base.
- GloVe is used when available. Otherwise it falls back to Word2Vec.
- The agent is retrieval-based, so it answers using the stored knowledge base rather than inventing new facts.
