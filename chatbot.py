import json
import os
import re
from ollama import chat

from retrieve import retrieve
from router import route, Intent, GREETINGS
from prompt_builder import build_dynamic_prompt
from db_manager import PatientMemoryManager
from consultation_manager import ConsultationState, ConsultationStage
from clinical_planner import ClinicalPlanner

MAX_HISTORY_MESSAGES = 10  # Turn on summarization when history reaches this limit

# Instantiate persistent SQLite EHR memory manager
memory_manager = PatientMemoryManager()


def load_profile():
    return memory_manager.get_patient_snapshot()


def save_profile(profile):
    memory_manager.update_patient_demographics(
        name=profile.get("name"),
        age=profile.get("age")
    )


def summarize_turns(messagesList):
    promptText = (
        "You are a medical conversation summarization engine. "
        "Your task is to summarize the following medical conversation snippet.\n\n"
        "INSTRUCTIONS:\n"
        "- Summarize the key medical symptoms, unresolved conditions, resolved conditions, and treatments discussed.\n"
        "- Preserve the medical chronology and context of what happened.\n"
        "- Keep the summary concise (1-2 sentences) and medically focused.\n"
        "- Never attempt to answer the user's questions or modify patient records. Output only the clinical summary.\n\n"
        "Snippet:\n"
    )
    for msg in messagesList:
        roleVal = "User" if msg["role"] == "user" else "Doctor"
        promptText += f"{roleVal}: {msg['content']}\n"

    try:
        resp = chat(
            model="phi4-mini:latest",
            messages=[{"role": "user", "content": promptText}]
        )
        return resp["message"]["content"].strip()
    except Exception as e:
        print(f"\n[Warning] History summarization failed: {e}")
        return ""


def merge_extracted_profile(current_profile, extracted_data):
    updated = False
    name_val = extracted_data.get("name")
    age_val = extracted_data.get("age")
    
    if (name_val and name_val != current_profile.get("name")) or \
       (age_val and age_val != current_profile.get("age")):
        memory_manager.update_patient_demographics(
            name=name_val or current_profile.get("name"),
            age=age_val or current_profile.get("age")
        )
        updated = True

    if updated:
        snapshot = memory_manager.get_patient_snapshot()
        current_profile.clear()
        current_profile.update(snapshot)

    return updated


def extract_profile_updates(user_msg, current_profile):
    prompt = (
        "You are a medical patient profile extractor.\n\n"
        "Your job is to read the latest user message, analyze it alongside the current profile, "
        "and determine any updates to the patient details: name and age.\n\n"
        "Return a clean JSON object containing ONLY the following keys:\n"
        "- name: string (null if not mentioned in the latest message)\n"
        "- age: integer (null if not mentioned in the latest message)\n\n"
        f"Current Patient Profile:\n{json.dumps(current_profile)}\n\n"
        f"Latest User Msg: \"{user_msg}\"\n\n"
        "Response JSON:"
    )
    try:
        resp = chat(
            model="phi4-mini:latest",
            messages=[{"role": "user", "content": prompt}],
            format="json"
        )
        new_data = json.loads(resp["message"]["content"])
        if not isinstance(new_data, dict):
            new_data = {}

        if merge_extracted_profile(current_profile, new_data):
            save_profile(current_profile)
    except Exception as e:
        print(f"\n[Warning] Profile extraction / DB update failed: {e}")


def extract_symptoms_in_memory(user_msg, consultation_state):
    userMsgClean = user_msg.lower().strip()
    extracted = []

    # 1. Direct "I have ..." or "I have some ..." or "I've got ..." or "I feel ..."
    haveMatch = re.search(r"\b(?:i\s+have|i've\s+got|i\s+am\s+experiencing|i'm\s+experiencing|i\s+feel|i'm\s+feeling|i\s+have\s+some|i\s+have\s+a|i\s+have\s+an)\s+([a-z0-9\s\-]+?)(?:\.|$|,|and\b|but\b|since\b|for\b)", userMsgClean)
    if haveMatch:
        candidate = haveMatch.group(1).strip()
        for prefix in ["a ", "some ", "an ", "mild ", "severe ", "moderate ", "concerning "]:
            if candidate.startswith(prefix):
                candidate = candidate[len(prefix):]
        candidate = re.split(r'\b(?:issues|problems|trouble|attacks|symptoms|since|for)\b', candidate)[0].strip()
        if candidate and candidate not in ["fine", "good", "better", "well", "okay", "ok", "great", "healthy", "fit", "normal", "nothing", "something", "no"]:
            extracted.append(candidate)

    # 2. Diagnosed / Suffering from
    diagMatch = re.search(r"\b(?:diagnosed\s+with|suffering\s+from)\s+(?:a\s+|an\s+|severe\s+|mild\s+)?([a-z0-9\s\-]+?)(?:\.|$|,|and\b|but\b)", userMsgClean)
    if diagMatch:
        candidate = diagMatch.group(1).strip()
        if candidate and candidate not in ["something", "anything"]:
            extracted.append(candidate)

    # 3. "My [body part/symptom] hurts" or "My [body part] pain"
    hurtMatch = re.search(r"\bmy\s+([a-z\s]+?)\s+hurts\b", userMsgClean)
    if hurtMatch:
        part = hurtMatch.group(1).strip()
        if part.endswith("pain") or part.endswith("ache"):
            extracted.append(part)
        elif part == "head":
            extracted.append("headache")
        elif part == "throat":
            extracted.append("sore throat")
        elif part == "stomach":
            extracted.append("stomach pain")
        elif part == "chest":
            extracted.append("chest pain")
        elif part == "back":
            extracted.append("back pain")
        elif part == "knee":
            extracted.append("knee pain")
        elif part == "neck":
            extracted.append("neck pain")
        elif part == "ear":
            extracted.append("ear pain")
        else:
            extracted.append(f"{part} pain")

    # 4. "I've been [symptom-ing]"
    ingMatch = re.search(r"\bi've\s+been\s+([a-z]+?ing)\b", userMsgClean)
    if ingMatch:
        verb = ingMatch.group(1).strip()
        if verb not in ["feeling", "doing", "having", "getting", "trying"]:
            if verb == "coughing":
                extracted.append("cough")
            elif verb == "vomiting":
                extracted.append("vomiting")
            elif verb == "sneezing":
                extracted.append("sneeze")
            elif verb == "bleeding":
                extracted.append("bleeding")
            else:
                base = verb[:-3]
                if base.endswith("t") and verb.endswith("tting"):
                    base = base[:-1]
                extracted.append(base)

    # 5. Specific terms
    negations = ["no ", "don't have", "dont have", "cured", "recovered", "resolved", "gone", "disappeared", "improved", "stopped", "fixed", "no longer", "no more"]
    
    if "fever" in userMsgClean and not any(neg in userMsgClean for neg in negations):
        tempMatch = re.search(r"\b(\d+(?:\.\d+)?\s*(?:°f|°c|f|c)?)\b", userMsgClean)
        tempVal = tempMatch.group(1) if tempMatch else None
        if tempVal and ("10" in tempVal or "9" in tempVal or "3" in tempVal):
            consultation_state.add_symptom("fever", {"temperature": tempVal})
        else:
            consultation_state.add_symptom("fever")
    if "stomach" in userMsgClean and ("hurt" in userMsgClean or "pain" in userMsgClean) and not any(neg in userMsgClean for neg in negations):
        extracted.append("stomach pain")
    if "throat" in userMsgClean and ("hurt" in userMsgClean or "sore" in userMsgClean) and not any(neg in userMsgClean for neg in negations):
        extracted.append("sore throat")

    # Add extracted symptoms to active session state
    for symptom in extracted:
        consultation_state.add_symptom(symptom)

    # Force resolution fallback: if the user explicitly says they are totally healthy/recovered/cured
    recovery_phrases = [
        "totally healthy", "completely healthy", "fully recovered", "back to normal",
        "completely cured", "totally fit", "fittest", "all cured", "feeling well",
        "totally fine", "completely fine", "recovered", "healthy now", "feeling good",
        "all my conditions are cured", "all my conditions have been cured", "no longer sick",
        "perfectly fine", "perfectly healthy", "fit now", "i m fine", "i'm fine", "i am fine",
        "healthy again", "doing great", "okay now", "symptom free", "feeling amazing",
        "absolutely fine", "perfectly alright"
    ]
    if any(phrase in userMsgClean for phrase in recovery_phrases):
        consultation_state.reset_session()


def main():
    messages = []
    last_intent = None
    history_summary = ""
    profile = load_profile()

    consultation_state = ConsultationState()
    planner = ClinicalPlanner()

    print("=" * 60)
    print("Doctor Chatbot Initialized!")
    print("Type 'exit' to end the conversation.")
    print("=" * 60)

    # Greet the user without blocking for name input on startup
    if not profile.get("name"):
        print("Doctor: Welcome! How can I assist you with your health today?")
    else:
        print(f"Doctor: Welcome back, {profile['name']}! How are you feeling today?")

    while True:
        try:
            question = input("\nYou : ").strip()
        except (KeyboardInterrupt, EOFError):
            print("\nGoodbye!")
            break

        if not question:
            continue

        textLower = question.lower()
        intent = route(question, last_intent, messages)

        if intent == Intent.EXIT:
            name = profile.get("name") or ""
            print(f"\nDoctor : Take care, {name}! Goodbye.")
            break

        # Determine whether patient database requires updating BEFORE generating response
        extract_profile_updates(question, profile)

        # Detect negative findings (explicitly denied symptoms)
        neg_patterns = [
            (r"\bno (cough|fever|sore throat|headache|chills|vomiting|nausea|diarrhea|rash|shortness of breath|chest pain)\b", 1),
            (r"\bdon't have (cough|fever|sore throat|headache|chills|vomiting|nausea|diarrhea|rash|shortness of breath|chest pain)\b", 1),
            (r"\bhaven't (?:had|noticed) (cough|fever|sore throat|headache|chills|vomiting|nausea|diarrhea|rash|shortness of breath|chest pain)\b", 1),
            (r"\bwithout (cough|fever|sore throat|headache|chills|vomiting|nausea|diarrhea|rash|shortness of breath|chest pain)\b", 1)
        ]
        for pat, group_idx in neg_patterns:
            m = re.search(pat, textLower)
            if m:
                consultation_state.add_negative_finding(m.group(group_idx))

        # Extract reported symptoms in-memory
        extract_symptoms_in_memory(question, consultation_state)

        # Run internal clinical planning step
        plan = planner.plan_turn(question, intent, consultation_state, profile)

        categoryFilter = None
        symptomNames = consultation_state.current_symptoms
        symptomsStr = " ".join(symptomNames).lower() if symptomNames else ""

        if any(w in textLower or w in symptomsStr for w in ["heart", "cardio", "ecg", "hypertension", "blood pressure"]):
            categoryFilter = "cardiology"
        elif any(w in textLower or w in symptomsStr for w in ["brain", "neurology", "headache", "stroke", "migraine", "tumor"]):
            categoryFilter = "neurology"
        elif any(w in textLower or w in symptomsStr for w in ["child", "pediatric", "baby", "infant", "kid"]):
            categoryFilter = "pediatrics"
        elif any(w in textLower or w in symptomsStr for w in ["cancer", "oncology", "chemo", "tumor", "malignant"]):
            categoryFilter = "oncology"
        elif any(w in textLower or w in symptomsStr for w in ["emergency", "accident", "trauma", "burn", "poisoning", "acute"]):
            categoryFilter = "emergency"

        # Conditional RAG: execute retrieval ONLY if clinical plan requires it
        documents = []
        if plan.requires_rag:
            ragQuery = question
            results = retrieve(ragQuery, category=categoryFilter, collection_name=plan.rag_collection)
            documents = results.get("documents", [])

        conversation = build_dynamic_prompt(
            question=question,
            intent=intent,
            plan=plan,
            consultation_state=consultation_state,
            profile=profile,
            documents=documents,
            historySummary=history_summary,
            messages=messages
        )

        print("\nDoctor : ", end="", flush=True)

        try:
            response_stream = chat(
                model="llama3.2:3b",
                messages=conversation,
                stream=True
            )

            assistant_reply = ""

            for chunk in response_stream:
                content = chunk["message"]["content"]
                assistant_reply += content
                print(content, end="", flush=True)

            print()

            # Record recommendations given to prevent repetition
            reply_lower = assistant_reply.lower()
            for adv_kw in ["hydrate", "hydration", "water", "rest", "sleep", "monitor"]:
                if adv_kw in reply_lower:
                    consultation_state.add_recommendation(adv_kw)

            # Update message history
            messages.append(
                {
                    "role": "user",
                    "content": question
                }
            )
            messages.append(
                {
                    "role": "assistant",
                    "content": assistant_reply
                }
            )

            # Dynamic history summarization
            if len(messages) >= MAX_HISTORY_MESSAGES:
                turns_to_summarize = messages[:4]  # take oldest 2 turns
                summary_chunk = summarize_turns(turns_to_summarize)
                if summary_chunk:
                    if history_summary:
                        history_summary += " " + summary_chunk
                    else:
                        history_summary = summary_chunk
                messages = messages[4:]  # prune summarized turns

            last_intent = intent

        except Exception as e:
            print(f"\n[Error] Ollama connection/inference failed: {e}")
            print("Please check that the Ollama service is running and 'llama3.2:3b' is pulled. 💀")


if __name__ == "__main__":
    main()