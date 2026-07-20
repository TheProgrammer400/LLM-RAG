import json
import os
from ollama import chat

from retrieve import retrieve
from router import route, Intent, GREETINGS
from prompt_builder import buildUnifiedPrompt
from disease_classifier import classify_disease
from db_manager import PatientMemoryManager

MAX_HISTORY_MESSAGES = 10  # Turn on summarization when history reaches this limit

# Instantiate persistent SQLite EHR memory manager
memory_manager = PatientMemoryManager()


def load_profile():
    return memory_manager.get_patient_snapshot()


def save_profile(profile):
    memory_manager.update_patient_demographics(
        name=profile.get("name"),
        age=profile.get("age"),
        gender=profile.get("gender")
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
    
    # Helper to ensure target is a list
    def get_list(d, key):
        val = d.get(key)
        if isinstance(val, list):
            return val
        if isinstance(val, str):
            return [val]
        return []

    # 1. Update name, age, and gender
    name_val = extracted_data.get("name")
    age_val = extracted_data.get("age")
    gender_val = extracted_data.get("gender")
    if (name_val and name_val != current_profile.get("name")) or \
       (age_val and age_val != current_profile.get("age")) or \
       (gender_val and gender_val != current_profile.get("gender")):
        memory_manager.update_patient_demographics(
            name=name_val or current_profile.get("name"),
            age=age_val or current_profile.get("age"),
            gender=gender_val or current_profile.get("gender")
        )
        updated = True

    # 2. Update list fields: allergies, medications, surgeries, family_history
    for item in get_list(extracted_data, "allergies"):
        item_clean = str(item).strip().lower()
        if item_clean and item_clean not in current_profile["allergies"]:
            memory_manager.add_allergy(allergy_name=item_clean)
            updated = True

    for item in get_list(extracted_data, "medications"):
        item_clean = str(item).strip().lower()
        if item_clean and item_clean not in current_profile["medications"]:
            memory_manager.add_medication(medication_name=item_clean)
            updated = True

    for item in get_list(extracted_data, "surgeries"):
        item_clean = str(item).strip().lower()
        if item_clean and item_clean not in current_profile["surgeries"]:
            memory_manager.add_surgery(surgery_name=item_clean)
            updated = True

    for item in get_list(extracted_data, "family_history"):
        item_clean = str(item).strip().lower()
        if item_clean and item_clean not in current_profile["family_history"]:
            memory_manager.add_family_history(detail=item_clean)
            updated = True

    # 3. Update chronic conditions
    for cond in get_list(extracted_data, "chronic_conditions"):
        if not isinstance(cond, dict):
            cond = {"name": str(cond).strip().lower()}
        c_name = cond.get("name", "").strip().lower()
        if not c_name:
            continue
        memory_manager.add_condition(condition_name=c_name, category="chronic", metadata=cond)
        updated = True

    # 4. Update new conditions (both suspected and active)
    new_active_list = get_list(extracted_data, "active_conditions")
    if new_active_list:
        for cond in new_active_list:
            if not isinstance(cond, dict):
                cond = {"name": str(cond).strip().lower()}
            c_name = cond.get("name", "").strip().lower()
            if not c_name:
                continue

            # Classify severity and check diagnostic confidence
            severity = classify_disease(c_name)
            is_confirmed = cond.get("confirmed", False)

            if severity == "MILD" or is_confirmed:
                memory_manager.add_condition(
                    condition_name=c_name,
                    category="active",
                    metadata=cond,
                    confidence="confirmed" if is_confirmed else "patient_reported",
                    severity=severity
                )
            else:
                memory_manager.add_condition(
                    condition_name=c_name,
                    category="suspected",
                    metadata=cond,
                    confidence="suspected",
                    severity=severity
                )
            updated = True

    # 5. Update resolved conditions (can resolve from active or suspected)
    new_resolved_list = get_list(extracted_data, "resolved_conditions")
    if new_resolved_list:
        for cond_name in new_resolved_list:
            cond_clean = str(cond_name).strip().lower()
            if cond_clean:
                memory_manager.resolve_condition(condition_name=cond_clean)
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
        "and determine any updates to the patient details: name, age, gender, allergies, chronic conditions, active conditions, resolved conditions, medications, surgeries, and family history.\n\n"
        "Return a clean JSON object containing ONLY the following keys:\n"
        "- name: string (null if not mentioned in the latest message)\n"
        "- age: integer (null if not mentioned in the latest message)\n"
        "- gender: string (null if not mentioned in the latest message)\n"
        "- allergies: list of new allergies (empty list if none)\n"
        "- chronic_conditions: list of chronic conditions reported. Each condition should be a structured object with a 'name' key (lowercase) and optional explicit metadata keys.\n"
        "- active_conditions: list of NEW active conditions (symptoms, illnesses, diseases, injuries) reported. Each condition MUST be a structured object containing a 'name' key (lowercase) and a 'confirmed' boolean key. Set 'confirmed' to true if the patient explicitly indicates they have been diagnosed with it by a doctor, are receiving medical treatment for it, or state it as a confirmed fact. Set 'confirmed' to false if the user only suspects, worries, or asks about having it (e.g. 'I think I have', 'I feel like I have', 'Maybe I have'). You must ONLY include other metadata keys (like 'temperature', 'blood_sugar', 'hba1c', etc.) if the user explicitly stated those metrics/values. DO NOT assume, default, or fabricate any other keys/values.\n"
        "- resolved_conditions: list of condition names (strings, lowercase) from the current profile that the user states are now cured, resolved, or gone. If the user states they are completely healthy/recovered, list all active condition names here.\n"
        "- medications: list of medications mentioned (empty list if none)\n"
        "- surgeries: list of surgeries mentioned (empty list if none)\n"
        "- family_history: list of family medical history details mentioned (empty list if none)\n\n"
        "CRITICAL RULES FOR RECOVERY STATEMENTS:\n"
        "- If the user indicates they have recovered, are completely healthy, fit, fine, back to normal, feeling well, or that all their conditions/symptoms are resolved (even without naming the conditions explicitly), you MUST identify all of the current active conditions from the profile and place their names in the 'resolved_conditions' list.\n\n"
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

        # Normalize and default confirmed key for active conditions
        if "active_conditions" in new_data and isinstance(new_data["active_conditions"], list):
            normalized_active = []
            for item in new_data["active_conditions"]:
                if isinstance(item, str):
                    c_name = item.strip().lower()
                    if c_name:
                        severity = classify_disease(c_name)
                        normalized_active.append({"name": c_name, "confirmed": True if severity == "MILD" else False})
                elif isinstance(item, dict):
                    c_name = item.get("name", "").strip().lower()
                    if c_name:
                        if "confirmed" not in item:
                            severity = classify_disease(c_name)
                            item["confirmed"] = True if severity == "MILD" else False
                        normalized_active.append(item)
            new_data["active_conditions"] = normalized_active
        
        # Backup deterministic python symptom/condition extractor
        userMsgClean = user_msg.lower()
        extractedConditions = []
        import re

        # 1. Direct "I have ..." or "I have some ..." or "I've got ..." or "I feel ..."
        haveMatch = re.search(r"\b(?:i\s+have|i've\s+got|i\s+am\s+experiencing|i'm\s+experiencing|i\s+feel|i'm\s+feeling|i\s+have\s+some|i\s+have\s+a|i\s+have\s+an)\s+([a-z0-9\s\-]+?)(?:\.|$|,|and\b|but\b|since\b|for\b)", userMsgClean)
        if haveMatch:
            candidate = haveMatch.group(1).strip()
            for prefix in ["a ", "some ", "an ", "mild ", "severe ", "moderate ", "concerning "]:
                if candidate.startswith(prefix):
                    candidate = candidate[len(prefix):]
            candidate = re.split(r'\b(?:issues|problems|trouble|attacks|symptoms|since|for)\b', candidate)[0].strip()
            if candidate and candidate not in ["fine", "good", "better", "well", "okay", "ok", "great", "healthy", "fit", "normal", "nothing", "something", "no"]:
                extractedConditions.append({"name": candidate})

        # 2. Diagnosed / Suffering from
        diagMatch = re.search(r"\b(?:diagnosed\s+with|suffering\s+from)\s+(?:a\s+|an\s+|severe\s+|mild\s+)?([a-z0-9\s\-]+?)(?:\.|$|,|and\b|but\b)", userMsgClean)
        if diagMatch:
            candidate = diagMatch.group(1).strip()
            if candidate and candidate not in ["something", "anything"]:
                extractedConditions.append({"name": candidate})

        # 3. "My [body part/symptom] hurts" or "My [body part] pain"
        hurtMatch = re.search(r"\bmy\s+([a-z\s]+?)\s+hurts\b", userMsgClean)
        if hurtMatch:
            part = hurtMatch.group(1).strip()
            if part.endswith("pain") or part.endswith("ache"):
                extractedConditions.append({"name": part})
            elif part == "head":
                extractedConditions.append({"name": "headache"})
            elif part == "throat":
                extractedConditions.append({"name": "sore throat"})
            elif part == "stomach":
                extractedConditions.append({"name": "stomach pain"})
            elif part == "chest":
                extractedConditions.append({"name": "chest pain"})
            elif part == "back":
                extractedConditions.append({"name": "back pain"})
            elif part == "knee":
                extractedConditions.append({"name": "knee pain"})
            elif part == "neck":
                extractedConditions.append({"name": "neck pain"})
            elif part == "ear":
                extractedConditions.append({"name": "ear pain"})
            else:
                extractedConditions.append({"name": f"{part} pain"})

        # 4. "I've been [symptom-ing]"
        ingMatch = re.search(r"\bi've\s+been\s+([a-z]+?ing)\b", userMsgClean)
        if ingMatch:
            verb = ingMatch.group(1).strip()
            if verb not in ["feeling", "doing", "having", "getting", "trying"]:
                if verb == "coughing":
                    extractedConditions.append({"name": "cough"})
                elif verb == "vomiting":
                    extractedConditions.append({"name": "vomiting"})
                elif verb == "sneezing":
                    extractedConditions.append({"name": "sneeze"})
                elif verb == "bleeding":
                    extractedConditions.append({"name": "bleeding"})
                else:
                    base = verb[:-3]
                    if base.endswith("t") and verb.endswith("tting"):
                        base = base[:-1]
                    extractedConditions.append({"name": base})

        # 5. Extract measurements if they are mentioned alongside terms
        negations = ["no ", "don't have", "dont have", "cured", "recovered", "resolved", "gone", "disappeared", "improved", "stopped", "fixed", "no longer", "no more"]
        
        if "fever" in userMsgClean and not any(neg in userMsgClean for neg in negations):
            tempMatch = re.search(r"\b(\d+(?:\.\d+)?\s*(?:°f|°c|f|c)?)\b", userMsgClean)
            tempVal = tempMatch.group(1) if tempMatch else None
            if tempVal and ("10" in tempVal or "9" in tempVal or "3" in tempVal):
                extractedConditions.append({"name": "fever", "temperature": tempVal})
            else:
                extractedConditions.append({"name": "fever"})

        if "stomach" in userMsgClean and ("hurt" in userMsgClean or "pain" in userMsgClean) and not any(neg in userMsgClean for neg in negations):
            extractedConditions.append({"name": "stomach pain"})

        if "throat" in userMsgClean and ("hurt" in userMsgClean or "sore" in userMsgClean) and not any(neg in userMsgClean for neg in negations):
            extractedConditions.append({"name": "sore throat"})

        if "blood sugar" in userMsgClean:
            sugarMatch = re.search(r"\b(\d+\s*(?:mg/dl|mg)?)\b", userMsgClean)
            if sugarMatch:
                extractedConditions.append({"name": "diabetes", "blood_sugar": sugarMatch.group(1)})

        if "blood pressure" in userMsgClean or "bp" in userMsgClean:
            bpMatch = re.search(r"\b(\d+/\d+)\b", userMsgClean)
            if bpMatch:
                extractedConditions.append({"name": "hypertension", "blood_pressure": bpMatch.group(1)})

        if "hba1c" in userMsgClean:
            hbaMatch = re.search(r"\b(\d+(?:\.\d+)?%)\b", userMsgClean)
            if hbaMatch:
                extractedConditions.append({"name": "diabetes", "hba1c": hbaMatch.group(1)})

        # Merge backup extractions into new_data["active_conditions"]
        if extractedConditions:
            if "active_conditions" not in new_data or not isinstance(new_data["active_conditions"], list):
                new_data["active_conditions"] = []
            
            # Determine diagnostic confirmation terms in user message
            confirmation_keywords = ["diagnosed", "confirm", "doctor", "test", "positive", "chemo", "radiation", "stage", "treatment", "hospitalized", "clinic", "physician", "have been taking", "prescribed"]
            is_confirmed_msg = any(ckw in userMsgClean for ckw in confirmation_keywords)
            
            for ec in extractedConditions:
                # Add default confirmed flag: True for mild, check keywords for moderate/severe
                severity = classify_disease(ec["name"])
                ec["confirmed"] = True if severity == "MILD" else is_confirmed_msg
                
                exists = False
                for c in new_data["active_conditions"]:
                    cName = c.get("name", "") if isinstance(c, dict) else str(c)
                    if cName.lower() == ec["name"]:
                        if isinstance(c, dict):
                            for k, v in ec.items():
                                c[k] = v
                        exists = True
                        break
                if not exists:
                    new_data["active_conditions"].append(ec)

        # Backup deterministic python condition resolver (scans active and suspected lists)
        for list_name in ["active_conditions", "suspected_conditions"]:
            for active in current_profile.get(list_name, []):
                aName = active.get("name", "") if isinstance(active, dict) else str(active)
                aNameClean = aName.lower().strip()
                if not aNameClean:
                    continue
                if aNameClean in userMsgClean:
                    recoveryKeywords = ["cured", "resolved", "gone", "disappeared", "stopped", "fixed", "improved", "no longer", "no more", "recovered", "healed", "fine now", "better now"]
                    if any(kw in userMsgClean for kw in recoveryKeywords):
                        if "resolved_conditions" not in new_data or not isinstance(new_data["resolved_conditions"], list):
                            new_data["resolved_conditions"] = []
                        if aNameClean not in new_data["resolved_conditions"]:
                            new_data["resolved_conditions"].append(aNameClean)

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
            hasBut = "but" in userMsgClean or "except" in userMsgClean
            stillSick = False
            if hasBut:
                for list_name in ["active_conditions", "suspected_conditions"]:
                    for active in current_profile.get(list_name, []):
                        a_name = active.get("name", "") if isinstance(active, dict) else str(active)
                        if a_name.lower() in userMsgClean:
                            stillSick = True
                            break
            if not stillSick:
                if "resolved_conditions" not in new_data or not isinstance(new_data["resolved_conditions"], list):
                    new_data["resolved_conditions"] = []
                for list_name in ["active_conditions", "suspected_conditions"]:
                    for active in current_profile.get(list_name, []):
                        a_name = active.get("name", "") if isinstance(active, dict) else str(active)
                        if a_name not in new_data["resolved_conditions"]:
                            new_data["resolved_conditions"].append(a_name)

        # Execute merging and validation using Python function
        if merge_extracted_profile(current_profile, new_data):
            save_profile(current_profile)
    except Exception as e:
        print(f"\n[Warning] Profile extraction / DB update failed: {e}")


def main():
    messages = []
    last_intent = None
    history_summary = memory_manager.get_latest_summary()
    profile = load_profile()

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

        intent = route(question, last_intent, messages)

        if intent == Intent.EXIT:
            name = profile.get("name") or ""
            print(f"\nDoctor : Take care, {name}! Goodbye.")
            break

        # Determine whether patient database requires updating BEFORE generating response
        extract_profile_updates(question, profile)

        categoryFilter = None
        textLower = question.lower()
        activeConditions = profile.get("active_conditions", [])
        suspectedConditions = profile.get("suspected_conditions", [])
        allConditions = activeConditions + suspectedConditions
        conditionNames = [c.get("name", "") if isinstance(c, dict) else str(c) for c in allConditions]
        profileConditionsStr = " ".join(conditionNames).lower() if conditionNames else ""

        if any(w in textLower or w in profileConditionsStr for w in ["heart", "cardio", "ecg", "hypertension", "blood pressure"]):
            categoryFilter = "cardiology"
        elif any(w in textLower or w in profileConditionsStr for w in ["brain", "neurology", "headache", "stroke", "migraine", "tumor"]):
            categoryFilter = "neurology"
        elif any(w in textLower or w in profileConditionsStr for w in ["child", "pediatric", "baby", "infant", "kid"]):
            categoryFilter = "pediatrics"
        elif any(w in textLower or w in profileConditionsStr for w in ["cancer", "oncology", "chemo", "tumor", "malignant"]):
            categoryFilter = "oncology"
        elif any(w in textLower or w in profileConditionsStr for w in ["emergency", "accident", "trauma", "burn", "poisoning", "acute"]):
            categoryFilter = "emergency"

        # Always retrieve documents if intent is MEDICAL or if the patient has active/suspected conditions
        documents = []
        if intent == Intent.MEDICAL or allConditions:
            ragQuery = question
            if intent != Intent.MEDICAL and conditionNames:
                # If the user is just chatting/greeting but has active conditions, fetch relevant documents for their conditions
                ragQuery = " ".join(conditionNames)
            
            results = retrieve(ragQuery, category=categoryFilter)
            documents = results.get("documents", [])

        conversation = buildUnifiedPrompt(
            question=question,
            messages=messages,
            documents=documents,
            profile=profile,
            historySummary=history_summary
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

            # Record consultation turn in SQLite database
            memory_manager.record_consultation(
                chief_complaint=question,
                assessment=assistant_reply,
                summary=history_summary
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
                    memory_manager.save_summary(summary=history_summary)
                messages = messages[4:]  # prune summarized turns

            last_intent = intent

        except Exception as e:
            print(f"\n[Error] Ollama connection/inference failed: {e}")
            print("Please check that the Ollama service is running and 'llama3.2:3b' is pulled. 💀")


if __name__ == "__main__":
    main()