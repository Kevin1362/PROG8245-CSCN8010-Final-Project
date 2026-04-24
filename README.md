# 🎓 Campus Support Agent

This project keeps the original **Campus Support Assistant** domain and upgrades it into a lightweight **agent** with **audio input**, **multilingual support**, and **spoken answers**.

## What the agent adds

- Intent detection for question types such as hours, location, contact, and procedures
- Short-term memory for follow-up questions such as **"Where is it?"**
- Automatic fallback guidance when confidence is low
- Choice of **Auto**, **GloVe**, or **Word2Vec** retrieval
- Gradio chat interface with retrieval details
- Audio input using microphone or uploaded audio
- Multilingual input and output
- Spoken answers using text-to-speech

## Project structure

```text
PROG8245-CSCN8010-Final-Project-Dry-Run/
│
├── app.py
├── README.md
├── requirements.txt
├── .gitignore
├── data/
│   └── knowledge_base.json
├── models/
│   └── word2vec_campus.bin
├── documentation/
│   └── architecture.md
├── notebooks/
│   └── Campus_Support_Assistant.ipynb
├── presentation/
│   └── campus_support_agent_presentation.pptx
└── src/
    ├── agent.py
    ├── tools.py
    ├── data_processing.py
    ├── evaluate.py
    ├── predict.py
    └── train.py
```

## How to run

### 1. Create and activate a virtual environment

**Windows PowerShell**

```powershell
py -3.11 -m venv .venv
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

## How the app works

- The user enters a question as text or audio
- If audio is used, the app converts speech to text
- The app detects the input language
- If needed, the question is translated into English
- The request is sent to the **CampusSupportAgent**
- The agent retrieves the best answer from the campus knowledge base
- The answer can be translated into the selected output language
- The answer can also be returned as spoken audio

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

## Architecture overview

- **Frontend:** Gradio UI, chatbot, text input, audio input, language selection, model selection, top-K slider
- **Middleware:** input handler, speech-to-text, language detection, translation, agent call, text-to-speech, response formatter
- **Backend:** CampusSupportAgent, intent detection, short-term memory, low-confidence fallback, retriever, model selector
- **Data and models:** knowledge base JSON, Word2Vec, GloVe embeddings
- **Optional services:** SpeechRecognition, deep-translator, gTTS

## Notes

- The project still uses your original campus FAQ knowledge base.
- GloVe is used when available. Otherwise it falls back to Word2Vec.
- The agent is retrieval-based, so it answers using the stored knowledge base rather than inventing new facts.
- Python **3.11** is recommended for best compatibility with `gensim`.
- Audio, translation, and spoken-answer features depend on the required packages being installed correctly.
