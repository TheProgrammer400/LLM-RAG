# Centralized disease severity classification database

MILD_KEYWORDS = [
    "cold", "common cold", "viral fever", "influenza", "flu", "seasonal allergies", "allergic rhinitis",
    "gastritis", "gerd", "acid reflux", "indigestion", "constipation", "gastroenteritis", "food poisoning",
    "dehydration", "migraine", "tension headache", "headache", "muscle strain", "back pain", "sprain", "sprains",
    "minor burns", "insect bites", "urinary tract infection", "uti", "conjunctivitis", "contact dermatitis",
    "eczema", "sore throat", "sinusitis", "cough", "burn"
]

MODERATE_KEYWORDS = [
    "diabetes", "hypertension", "high blood pressure", "asthma", "bronchitis", "pneumonia", "dengue",
    "malaria", "typhoid", "kidney stone", "kidney stones", "gallstone", "gallstones", "hypothyroidism",
    "hyperthyroidism", "osteoarthritis", "rheumatoid arthritis", "arthritis", "psoriasis", "epilepsy",
    "pcos", "endometriosis", "sleep apnea", "chronic sinusitis", "peptic ulcer", "ulcer"
]

SEVERE_KEYWORDS = [
    "cancer", "hiv", "aids", "hepatitis", "rabies", "tuberculosis", "tb", "heart attack", "stroke",
    "heart failure", "multiple sclerosis", "parkinson", "als", "kidney failure", "renal failure",
    "kidney disease", "renal disease", "liver failure", "hepatic failure", "cirrhosis", "lupus",
    "vasculitis", "scleroderma", "organ failure", "transplant rejection", "leukemia", "lymphoma",
    "sarcoma", "melanoma", "carcinoma", "meningitis", "sepsis", "septic shock", "pulmonary embolism",
    "aortic dissection", "hemorrhage", "aneurysm", "brain hemorrhage", "brain aneurysm"
]


def classify_disease(diseaseName):
    """
    Classifies a disease into MILD, MODERATE, or SEVERE.
    If not recognized, defaults to MODERATE to require confirmation.
    """
    nameClean = str(diseaseName).lower().strip()
    
    # Check severe first
    for keyword in SEVERE_KEYWORDS:
        if keyword in nameClean:
            return "SEVERE"
            
    # Check moderate
    for keyword in MODERATE_KEYWORDS:
        if keyword in nameClean:
            return "MODERATE"
            
    # Check mild
    for keyword in MILD_KEYWORDS:
        if keyword in nameClean:
            return "MILD"
            
    # Safe default: treat as MODERATE to enforce confirmation
    return "MODERATE"


def get_severity(diseaseName):
    """
    Alias of classify_disease to comply with interface requirements.
    """
    return classify_disease(diseaseName)
