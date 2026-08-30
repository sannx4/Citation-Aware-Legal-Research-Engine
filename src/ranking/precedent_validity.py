NEGATIVE_TREATMENTS = {
    "OVERRULED": 0.60,
    "DOUBTED": 0.35,
    "CRITICIZED": 0.25,
    "DISTINGUISHED": 0.15,
}

POSITIVE_TREATMENTS = {
    "FOLLOWED": 0.10,
    "RELIED_ON": 0.10,
    "APPROVED": 0.08,
    "REFERRED": 0.03,
}


def negative_treatment_penalty(treatments: list[str]) -> float:
    penalty = 0.0

    for treatment in treatments:
        treatment = str(treatment).upper().strip()
        penalty += NEGATIVE_TREATMENTS.get(treatment, 0.0)

    return min(penalty, 0.90)


def positive_treatment_bonus(treatments: list[str]) -> float:
    bonus = 0.0

    for treatment in treatments:
        treatment = str(treatment).upper().strip()
        bonus += POSITIVE_TREATMENTS.get(treatment, 0.0)

    return min(bonus, 0.30)