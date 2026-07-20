from enum import Enum
import re


class Intent(Enum):
    EXIT = 0
    GREETING = 1
    SMALL_TALK = 2
    MEDICAL = 3
    CONVERSATION = 4


GREETINGS = {
    "hi",
    "hello",
    "hey",
    "good morning",
    "good afternoon",
    "good evening",
    "yo",
    "sup"
}

SMALL_TALK = {
    "ok",
    "okay",
    "thanks",
    "thank you",
    "alright",
    "alright then",
    "cool",
    "nice",
    "great",
    "awesome",
    "bye",
    "goodbye",
    "see you",
    "good night"
}

MEDICAL_KEYWORDS = {
    "fever",
    "cold",
    "cough",
    "flu",
    "headache",
    "migraine",
    "pain",
    "ache",
    "vomit",
    "vomiting",
    "nausea",
    "diarrhea",
    "constipation",
    "dizziness",
    "fatigue",
    "weakness",
    "infection",
    "virus",
    "bacteria",
    "medicine",
    "medication",
    "tablet",
    "capsule",
    "pill",
    "drug",
    "prescription",
    "treatment",
    "diagnosis",
    "symptom",
    "doctor",
    "hospital",
    "brain",
    "tumor",
    "stroke",
    "hypertension",
    "blood pressure",
    "bp",
    "diabetes",
    "heart",
    "chest",
    "ecg",
    "mri",
    "ct",
    "scan",
    "xray",
    "fracture",
    "injury",
    "bleeding",
    "rash",
    "asthma",
    "breathing",
    "seizure",
    "paralysis",
    "swelling"
}

MEDICAL_PATTERNS = {

    r"\bi have\b",
    r"\bi've\b",
    r"\bi am having\b",
    r"\bi'm having\b",

    r"\bi feel\b",
    r"\bi'm feeling\b",

    r"\bmy .+ hurts\b",
    r"\bmy .+ is hurting\b",

    r"\bi am suffering from\b",
    r"\bi'm suffering from\b",

    r"\bi think i have\b",

    r"\brecommend\b",
    r"\bmedicine\b",
    r"\bmedication\b",

    r"\bwhat medicine\b",
    r"\bwhich medicine\b",
    r"\bwhat tablet\b",
    r"\bwhich tablet\b",

    r"\bhow to treat\b",
    r"\bhow do i treat\b",

    r"\bwhat causes\b",

    r"\bsymptoms of\b",

    r"\btreatment for\b",

    r"\bis .* dangerous\b",

    r"\bshould i see a doctor\b",

    r"\bhigh fever\b",

    r"\blow fever\b",

    r"\bbody pain\b",

    r"\bchest pain\b",

    r"\bshortness of breath\b",

    r"\bdifficulty breathing\b"
}


from ollama import chat

FOLLOW_UP_WORDS = {
    "why", "how", "what", "who", "which", "does", "is", "can", "should", "will",
    "explain", "tell", "more", "yes", "no", "okay", "thanks", "thank"
}


def route_llm(question: str, history: list = None) -> Intent:
    chat_context = ""
    if history:
        # Get the last 4 turns (2 user, 2 assistant) to provide context for follow-up questions
        recent = history[-4:]
        chat_context = "Recent conversation turns:\n"
        for msg in recent:
            role = "User" if msg["role"] == "user" else "Doctor"
            chat_context += f"{role}: {msg['content']}\n"
        chat_context += "\n"

    prompt = (
        "You are a medical chatbot router. Your job is to classify the user's latest query into one of three intents based on the question and optional conversation context:\n"
        "- EXIT: If the user wants to end the chat (e.g., 'exit', 'bye', 'goodbye', 'quit', 'talk to you later').\n"
        "- MEDICAL: If the user is asking about symptoms, diseases, medical advice, drug info, ECG, treatments, or asking follow-up questions to their current health issues (e.g. 'why?', 'what should I do?', 'is it dangerous?' following a medical symptom).\n"
        "- CONVERSATION: If the user is greeting you, doing small talk, asking general questions (e.g. 'what is my name', 'who are you', 'how are you'), or chatting about off-topic items.\n\n"
        f"{chat_context}"
        f"User Query: \"{question}\"\n\n"
        "Respond with ONLY one of these exact words in uppercase: EXIT, MEDICAL, CONVERSATION. Do not write any other explanation or punctuation."
    )

    try:
        resp = chat(
            model="llama3.2:3b",
            messages=[
                {
                    "role": "user",
                    "content": prompt
                }
            ]
        )
        output = resp["message"]["content"].strip().upper()
        # Clean up punctuation
        output = "".join(c for c in output if c.isalnum())

        if "EXIT" in output:
            return Intent.EXIT
        elif "MEDICAL" in output:
            return Intent.MEDICAL
        elif "CONVERSATION" in output or "GREETING" in output or "SMALL_TALK" in output:
            return Intent.CONVERSATION
    except Exception as e:
        print(f"\n[Warning] Semantic LLM routing failed: {e}. Falling back to rule-based routing.")

    return None


def route(question: str, last_intent: Intent = None, history: list = None) -> Intent:
    # 1. Try Fast Rule-based Routing first to save Ollama API latency
    text = question.strip().lower()
    clean_text = re.sub(r"[.!?,\s]+", "", text)

    if clean_text == "exit":
        return Intent.EXIT

    if clean_text in GREETINGS:
        return Intent.GREETING

    if clean_text in SMALL_TALK:
        return Intent.SMALL_TALK

    for pattern in MEDICAL_PATTERNS:
        if re.search(pattern, text):
            return Intent.MEDICAL

    for keyword in MEDICAL_KEYWORDS:
        if re.search(rf"\b{re.escape(keyword)}\b", text):
            return Intent.MEDICAL

    # Context-aware follow-up check:
    # If last intent was medical, classify follow-up query as medical.
    if last_intent == Intent.MEDICAL:
        words = text.split()
        if len(words) > 0:
            first_word = words[0].strip("?,.!")
            if first_word in FOLLOW_UP_WORDS:
                return Intent.MEDICAL
            if any(pronoun in words for pronoun in ["it", "this", "that", "they", "them", "those"]):
                return Intent.MEDICAL
            if len(words) <= 4:
                return Intent.MEDICAL

    # 2. If rules fail to categorize, fallback to Semantic LLM Router
    llm_intent = route_llm(question, history)
    if llm_intent is not None:
        return llm_intent

    return Intent.CONVERSATION