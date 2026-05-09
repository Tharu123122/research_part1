from dataclasses import dataclass


@dataclass
class RiskResult:
    score: float
    level: str
    reasons: list[str]
    recommendations: list[str]


def yes(value: object) -> bool:
    return str(value).strip().lower() in {"yes", "true", "1", "y"}


def level_from_score(score: float) -> str:
    if score < 35:
        return "Low"
    if score < 65:
        return "Moderate"
    return "High"


def calculate_preventive_risk(user_data: dict) -> RiskResult:
    score = 0.0
    reasons = []
    recommendations = []

    age = float(user_data.get("Age", 0) or 0)
    if age >= 60:
        score += 12
        reasons.append("Age is above 60, which increases oral cancer risk.")
    elif age >= 45:
        score += 7
        reasons.append("Age is above 45, so regular oral screening is recommended.")

    risk_rules = [
        ("Tobacco Use", 18, "Tobacco use is a major oral cancer risk factor.", "Reduce or stop tobacco use and seek cessation support."),
        ("Alcohol Consumption", 12, "Alcohol consumption increases oral cancer risk.", "Reduce alcohol consumption and avoid combining alcohol with tobacco."),
        ("Betel Quid Use", 20, "Betel quid chewing is strongly associated with oral cancer risk.", "Stop betel quid chewing and attend oral screening."),
        ("Poor Oral Hygiene", 10, "Poor oral hygiene can increase oral disease risk.", "Improve brushing, flossing, and dental checkups."),
        ("HPV Infection", 8, "HPV infection is associated with some oral and throat cancers.", "Discuss HPV-related risk and vaccination/screening with a clinician."),
        ("Family History of Cancer", 7, "Family history may increase cancer risk.", "Schedule regular oral health examinations."),
        ("Compromised Immune System", 7, "A weakened immune system can increase health risk.", "Seek medical advice for regular monitoring."),
        ("Oral Lesions", 15, "Oral lesions are an important warning indicator.", "If lesions persist for more than two weeks, consult a dentist or doctor."),
        ("Unexplained Bleeding", 12, "Unexplained oral bleeding is a warning sign.", "Seek clinical examination for unexplained bleeding."),
        ("Difficulty Swallowing", 12, "Difficulty swallowing can be a concerning symptom.", "Consult a healthcare professional for evaluation."),
        ("White or Red Patches in Mouth", 15, "White or red patches may require clinical assessment.", "Arrange an oral screening if patches persist."),
    ]

    for field, points, reason, recommendation in risk_rules:
        if yes(user_data.get(field)):
            score += points
            reasons.append(reason)
            recommendations.append(recommendation)

    diet = str(user_data.get("Diet (Fruits & Vegetables Intake)", "")).strip().lower()
    if diet == "low":
        score += 6
        reasons.append("Low fruit and vegetable intake may reduce protective dietary benefits.")
        recommendations.append("Increase fruit and vegetable intake as part of a healthier diet.")

    if yes(user_data.get("Tobacco Use")) and yes(user_data.get("Alcohol Consumption")):
        score += 10
        reasons.append("Combined tobacco and alcohol use creates a higher-risk habit pattern.")
        recommendations.append("Prioritize reducing tobacco and alcohol together.")

    if yes(user_data.get("Betel Quid Use")) and yes(user_data.get("Poor Oral Hygiene")):
        score += 8
        reasons.append("Betel quid use combined with poor oral hygiene increases concern.")
        recommendations.append("Stop betel chewing and improve oral hygiene immediately.")

    symptom_fields = [
        "Oral Lesions",
        "Unexplained Bleeding",
        "Difficulty Swallowing",
        "White or Red Patches in Mouth",
    ]
    symptom_count = sum(yes(user_data.get(field)) for field in symptom_fields)
    if symptom_count >= 2:
        score += 10
        reasons.append("Multiple oral warning symptoms are present.")
        recommendations.append("Seek professional oral examination as soon as possible.")

    score = min(score, 100.0)
    if not recommendations:
        recommendations.append("Maintain good oral hygiene and continue regular dental checkups.")

    return RiskResult(
        score=round(score, 2),
        level=level_from_score(score),
        reasons=reasons or ["No major lifestyle or symptom risk factors were reported."],
        recommendations=list(dict.fromkeys(recommendations)),
    )
