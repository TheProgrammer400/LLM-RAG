import json
from prompts import BASE_SYSTEM_PROMPT


def buildUnifiedPrompt(question, messages, documents, profile=None, historySummary=None):
    # 1. Format the Patient Profile (with extensible fields)
    profileStr = ""
    if profile:
        profileStr = "========================\nPATIENT PROFILE\n========================\n"
        if profile.get("name"):
            profileStr += f"- Name: {profile['name']}\n"
        if profile.get("age"):
            profileStr += f"- Age: {profile['age']}\n"
        if profile.get("gender"):
            profileStr += f"- Gender: {profile['gender']}\n"
        
        # Format list-based profile arrays
        for listKey in ["allergies", "medications", "surgeries", "family_history"]:
            valList = profile.get(listKey, [])
            if valList:
                profileStr += f"- {listKey.replace('_', ' ').capitalize()}: {', '.join(valList)}\n"
                
        # Format condition-based profile arrays (dicts or strings)
        for condKey in ["chronic_conditions", "active_conditions", "suspected_conditions", "resolved_conditions"]:
            valList = profile.get(condKey, [])
            if valList:
                profileStr += f"- {condKey.replace('_', ' ').capitalize()}:\n"
                for item in valList:
                    if isinstance(item, dict):
                        nameVal = item.get("name", "unknown")
                        metaVal = ", ".join(f"{mk}: {mv}" for mk, mv in item.items() if mk != "name")
                        profileStr += f"  * {nameVal} ({metaVal})\n" if metaVal else f"  * {nameVal}\n"
                    else:
                        profileStr += f"  * {item}\n"
        profileStr += "\n"

    # 2. Format the Conversation Summary
    summaryStr = ""
    if historySummary:
        summaryStr = (
            "========================\nCONVERSATION SUMMARY\n========================\n"
            f"{historySummary}\n\n"
        )

    # 3. Format the Retrieved Medical Knowledge (RAG)
    referenceStr = ""
    if documents:
        contextVal = "\n\n------------------------------\n\n".join(documents)
        referenceStr = (
            "========================\nRETRIEVED MEDICAL KNOWLEDGE (RAG)\n========================\n"
            "Relevant medical reference found. Base your answer primarily on the retrieved context.\n"
            "If the retrieved documents contain only partial information or if there are gaps in the retrieved context regarding the user's specific query, "
            "you must seamlessly fill the missing gaps using your own general medical knowledge to provide a complete, comprehensive, and high-quality response.\n"
            "Never immediately refuse or state that you lack information.\n\n"
            f"Reference Context:\n{contextVal}\n\n"
        )
    else:
        referenceStr = (
            "========================\nRETRIEVED MEDICAL KNOWLEDGE (RAG)\n========================\n"
            "No specific reference textbook documents were retrieved for this query. Answer directly using your general medical knowledge.\n"
            "Provide a complete, comprehensive, medically accurate, and detailed response.\n"
            "NEVER refuse safe educational medical questions, and NEVER state that you cannot provide medical information or that the knowledge base lacks information on this topic.\n\n"
        )

    # 4. Refusal and greeting rules
    rulesStr = (
        "========================\nADDITIONAL RULES\n========================\n"
        "1. Off-Topic Refusals: Your role is strictly and completely limited to medical and health-related topics. "
        "If the user asks about external non-medical off-topic subjects (such as general trivia, coding, history, celebrities, movies, pop culture, etc.), "
        "you must IMMEDIATELY and POLITELY DECLINE, state that you can only help with medical or health-related topics, and prompt them to ask a health-related question. Do NOT provide any external facts or trivia about the off-topic subject under any circumstances.\n"
        "2. Greetings & Meta-Questions: If the user greets you or asks about the conversation itself (like asking for their name, their age, their profile details, or who you are), answer politely in character as the Physician Assistant. E.g., if they ask for their name or age, tell them what is recorded in the Patient Profile, or say you don't know it yet if it is not in the Patient Profile.\n"
        "3. Never refer to sources, document names, or PDFs. Never say 'according to the documents'. Speak naturally like an experienced physician assistant.\n"
        "4. CRITICAL DISTINCTION: Do NOT confuse RETRIEVED MEDICAL KNOWLEDGE (reference textbooks) with the PATIENT PROFILE. The retrieved reference context contains general medical textbook knowledge for your clinical guidance; it is NOT the patient's personal medical history. The patient's actual medical history, reported symptoms, and profile are strictly listed under the PATIENT PROFILE and CONVERSATION SUMMARY sections. Only attribute symptoms/conditions to the patient if they appear in the PATIENT PROFILE or CONVERSATION SUMMARY.\n"
        "5. Educational Medical Questions: Questions asking about diseases, causes, symptoms, risk factors, diagnostic tests, or treatments are requests for safe medical education, NOT requests for a personal diagnosis. Answer all educational medical questions directly, thoroughly, and in detail without stating that you cannot provide information.\n"
    )

    # Context Construction Order:
    # 1. System Prompt
    # 2. Patient Profile
    # 3. Conversation Summary
    # 4. Retrieved Medical Knowledge (RAG)
    # Rules are appended at the end of the system block for instruction enforcement.
    systemContent = (
        f"{BASE_SYSTEM_PROMPT}\n\n"
        f"{profileStr}"
        f"{summaryStr}"
        f"{referenceStr}"
        f"{rulesStr}"
    )

    conversationList = [
        {
            "role": "system",
            "content": systemContent
        }
    ]
    # 5. Recent Conversation
    conversationList.extend(messages)
    # 6. Latest User Message
    conversationList.append(
        {
            "role": "user",
            "content": question
        }
    )
    return conversationList