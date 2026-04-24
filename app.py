"""
app.py
------
Gradio web application for the Campus Support Agent.

Upgrades added:
- Audio input (microphone or upload)
- Multilingual input/output
- Spoken answers (text-to-speech)
- Keeps CampusSupportAgent as the core answer engine

Launch
------
    python app.py

Then open http://localhost:7860 in your browser.
"""

from __future__ import annotations

import sys
import tempfile
from pathlib import Path
from typing import Optional

import gradio as gr

# Ensure src/ is on the path when running from the project root
sys.path.insert(0, str(Path(__file__).parent / "src"))

from agent import CampusSupportAgent

# Optional libraries for multilingual + audio features
try:
    import speech_recognition as sr
except Exception:
    sr = None

try:
    from deep_translator import GoogleTranslator
except Exception:
    GoogleTranslator = None

try:
    from gtts import gTTS
except Exception:
    gTTS = None

try:
    from langdetect import detect
except Exception:
    detect = None


AGENT = CampusSupportAgent(default_top_k=3)

# ---------------------------------------------------------------------------
# Language settings
# ---------------------------------------------------------------------------

INPUT_LANGUAGE_CODES = {
    "Auto Detect": "auto",
    "English": "en",
    "French": "fr",
    "Spanish": "es",
    "Hindi": "hi",
    "Gujarati": "gu",
}

OUTPUT_LANGUAGE_CODES = {
    "English": "en",
    "French": "fr",
    "Spanish": "es",
    "Hindi": "hi",
    "Gujarati": "gu",
}

SPEECH_RECOGNITION_LOCALES = {
    "English": ["en-CA", "en-US", "en-GB"],
    "French": ["fr-CA", "fr-FR"],
    "Spanish": ["es-ES", "es-MX"],
    "Hindi": ["hi-IN"],
    "Gujarati": ["gu-IN", "hi-IN", "en-CA"],
    "Auto Detect": ["en-CA", "en-US", "fr-CA", "fr-FR", "es-ES", "hi-IN", "gu-IN"],
}

GTTS_LANGUAGE_CODES = {"en", "fr", "es", "hi", "gu"}
LANGUAGE_LABEL_BY_CODE = {
    code: label for label, code in INPUT_LANGUAGE_CODES.items() if code != "auto"
}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def detect_input_language(text: str, selected_language: str) -> str:
    """Return the detected or selected input language code."""
    if selected_language != "Auto Detect":
        return INPUT_LANGUAGE_CODES[selected_language]

    if detect is None:
        return "en"

    try:
        detected = detect(text)
        return detected if detected in LANGUAGE_LABEL_BY_CODE else "en"
    except Exception:
        return "en"



def translate_text(text: str, source_lang: str, target_lang: str) -> str:
    """Translate text if translation is available, otherwise return original."""
    if not text.strip() or source_lang == target_lang:
        return text

    if GoogleTranslator is None:
        return text

    try:
        translated = GoogleTranslator(source=source_lang, target=target_lang).translate(text)
        return translated if translated else text
    except Exception:
        return text



def transcribe_audio(audio_path: Optional[str], input_language: str) -> tuple[str, str]:
    """Convert audio file to text using SpeechRecognition."""
    if not audio_path:
        return "", "No audio input provided."

    if sr is None:
        return "", "Audio-to-text is unavailable because speech_recognition is not installed."

    recognizer = sr.Recognizer()

    try:
        with sr.AudioFile(audio_path) as source:
            audio_data = recognizer.record(source)
    except Exception as exc:
        return "", f"Could not read audio file: {exc}"

    locales = SPEECH_RECOGNITION_LOCALES.get(input_language, ["en-CA"])

    for locale in locales:
        try:
            transcript = recognizer.recognize_google(audio_data, language=locale)
            return transcript, f"Audio transcribed successfully using locale {locale}."
        except sr.UnknownValueError:
            continue
        except sr.RequestError as exc:
            return "", f"Speech recognition service error: {exc}"
        except Exception:
            continue

    return "", "Could not understand the audio. Please try again or type your question."



def synthesize_audio(text: str, output_language_code: str) -> tuple[Optional[str], str]:
    """Convert answer text to speech."""
    if not text.strip():
        return None, "No answer text available for audio output."

    if gTTS is None:
        return None, "Spoken answer is unavailable because gTTS is not installed."

    tts_language = output_language_code if output_language_code in GTTS_LANGUAGE_CODES else "en"

    try:
        temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".mp3")
        temp_file.close()
        gTTS(text=text, lang=tts_language, slow=False).save(temp_file.name)
        return temp_file.name, f"Spoken answer generated in '{tts_language}'."
    except Exception as exc:
        return None, f"Could not generate spoken answer: {exc}"



def build_status_text(messages: list[str]) -> str:
    return "\n".join(message for message in messages if message)


# ---------------------------------------------------------------------------
# Main handler
# ---------------------------------------------------------------------------


def chat_with_agent(
    query: str,
    audio_path: Optional[str],
    chat_history: list[dict] | None,
    session_state: dict | None,
    input_language: str,
    output_language: str,
    model_choice: str,
    top_k: int,
):
    """Main Gradio handler for the Campus Support Agent with voice + multilingual support."""
    chat_history = chat_history or []
    session_state = session_state or {"last_subject": None, "turn_count": 0, "last_intent": None}
    status_messages: list[str] = []
    transcript_text = ""

    query = (query or "").strip()

    if query:
        user_text = query
        status_messages.append("Using typed text input.")
    else:
        user_text, audio_status = transcribe_audio(audio_path, input_language)
        transcript_text = user_text
        status_messages.append(audio_status)

    if not user_text.strip():
        return (
            chat_history,
            session_state,
            "",
            "",
            transcript_text,
            "",
            build_status_text(status_messages + ["Please enter a question or record audio."]),
            None,
        )

    detected_language_code = detect_input_language(user_text, input_language)
    detected_language_label = LANGUAGE_LABEL_BY_CODE.get(detected_language_code, "English")
    status_messages.append(
        f"Input language used: {detected_language_label} ({detected_language_code})."
    )

    retrieval_query = user_text
    if detected_language_code != "en":
        translated_query = translate_text(user_text, detected_language_code, "en")
        if translated_query != user_text:
            retrieval_query = translated_query
            status_messages.append("Translated input to English for the agent.")
        else:
            status_messages.append(
                "Translation service unavailable, so the agent used the original text."
            )

    result = AGENT.answer(
        query=retrieval_query,
        session_state=session_state,
        model_choice=model_choice,
        top_k=int(top_k),
    )

    answer_en = result.get("answer", "No answer found.")
    details_en = AGENT.format_details(result)

    output_language_code = OUTPUT_LANGUAGE_CODES[output_language]
    answer_out = answer_en
    details_out = details_en

    if output_language_code != "en":
        translated_answer = translate_text(answer_en, "en", output_language_code)
        translated_details = translate_text(details_en, "en", output_language_code)

        if translated_answer != answer_en:
            answer_out = translated_answer
            details_out = translated_details
            status_messages.append(f"Translated answer to {output_language}.")
        else:
            status_messages.append(
                "Could not translate the answer, so English output is shown."
            )

    chat_history.append({"role": "user", "content": user_text})
    chat_history.append({"role": "assistant", "content": answer_out})

    session_state["last_subject"] = result.get("subject")
    session_state["last_intent"] = result.get("intent")
    session_state["turn_count"] = int(session_state.get("turn_count", 0)) + 1

    audio_file, audio_status = synthesize_audio(answer_out, output_language_code)
    status_messages.append(audio_status)

    return (
        chat_history,
        session_state,
        "",
        details_out,
        transcript_text,
        retrieval_query,
        build_status_text(status_messages),
        audio_file,
    )



def clear_all():
    return [], {"last_subject": None, "turn_count": 0, "last_intent": None}, "", "", "", "", "", None


# ---------------------------------------------------------------------------
# UI
# ---------------------------------------------------------------------------

with gr.Blocks(title="Campus Support Agent") as demo:
    gr.Markdown(
        """
        # 🎓 Campus Support Agent
        Ask campus-related questions with **text or voice**.

        This upgraded version keeps your **CampusSupportAgent** and adds:
        - 🎤 Audio input
        - 🌍 Multilingual input/output
        - 🔊 Spoken answers
        - 🤖 Same agent-based chat flow
        """
    )

    session_state = gr.State({"last_subject": None, "turn_count": 0, "last_intent": None})

    chatbot = gr.Chatbot(label="Conversation", height=420, type="messages")

    with gr.Row():
        with gr.Column(scale=3):
            query_box = gr.Textbox(
                label="Your question",
                placeholder="Type a question or leave this blank and use the microphone.",
                lines=2,
            )
            audio_input = gr.Audio(
                sources=["microphone", "upload"],
                type="filepath",
                format="wav",
                label="Record or upload your question as audio",
            )
        with gr.Column(scale=1):
            input_lang_dropdown = gr.Dropdown(
                choices=list(INPUT_LANGUAGE_CODES.keys()),
                value="Auto Detect",
                label="Input language",
            )
            output_lang_dropdown = gr.Dropdown(
                choices=list(OUTPUT_LANGUAGE_CODES.keys()),
                value="English",
                label="Output language",
            )
            model_radio = gr.Radio(
                choices=["Auto", "GloVe", "Word2Vec"],
                value="Auto",
                label="Embedding model",
            )
            top_k_slider = gr.Slider(
                minimum=1,
                maximum=5,
                step=1,
                value=3,
                label="Top-K results",
            )

    with gr.Row():
        ask_btn = gr.Button("Ask", variant="primary")
        clear_btn = gr.Button("Clear")

    with gr.Row():
        with gr.Column():
            answer_audio = gr.Audio(label="Spoken answer", type="filepath", interactive=False)
        with gr.Column():
            transcript_box = gr.Textbox(label="Audio transcript", lines=3, interactive=False)
            retrieval_query_box = gr.Textbox(
                label="Query sent to the agent",
                lines=3,
                interactive=False,
            )
            status_box = gr.Textbox(label="Status", lines=6, interactive=False)

    details_box = gr.Markdown(label="Agent details")

    gr.Examples(
        examples=[
            ["What time does the library open?", None, "Auto Detect", "English", "Auto", 3],
            ["Where is the financial aid office?", None, "Auto Detect", "French", "Auto", 3],
            ["How can I contact student services?", None, "Auto Detect", "Hindi", "Auto", 3],
            ["Is there a gym on campus?", None, "Auto Detect", "Gujarati", "Auto", 3],
            ["Where is it?", None, "Auto Detect", "English", "Auto", 3],
        ],
        inputs=[
            query_box,
            audio_input,
            input_lang_dropdown,
            output_lang_dropdown,
            model_radio,
            top_k_slider,
        ],
    )

    ask_btn.click(
        fn=chat_with_agent,
        inputs=[
            query_box,
            audio_input,
            chatbot,
            session_state,
            input_lang_dropdown,
            output_lang_dropdown,
            model_radio,
            top_k_slider,
        ],
        outputs=[
            chatbot,
            session_state,
            query_box,
            details_box,
            transcript_box,
            retrieval_query_box,
            status_box,
            answer_audio,
        ],
    )

    query_box.submit(
        fn=chat_with_agent,
        inputs=[
            query_box,
            audio_input,
            chatbot,
            session_state,
            input_lang_dropdown,
            output_lang_dropdown,
            model_radio,
            top_k_slider,
        ],
        outputs=[
            chatbot,
            session_state,
            query_box,
            details_box,
            transcript_box,
            retrieval_query_box,
            status_box,
            answer_audio,
        ],
    )

    clear_btn.click(
        fn=clear_all,
        inputs=None,
        outputs=[
            chatbot,
            session_state,
            query_box,
            details_box,
            transcript_box,
            retrieval_query_box,
            status_box,
            answer_audio,
        ],
    )

    gr.Markdown(
        """
        ---
        ### What this version does
        1. Accepts typed text or microphone/uploaded audio.
        2. Detects or uses the selected input language.
        3. Translates non-English input to English for your agent.
        4. Sends the query into `CampusSupportAgent`.
        5. Translates the answer to the selected output language.
        6. Generates a spoken answer when text-to-speech is available.

        Optional packages for these features:
        - speechrecognition
        - deep-translator
        - gtts
        - langdetect

        If any optional package is missing, the app still runs and falls back gracefully.
        """
    )


if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860, share=True)
