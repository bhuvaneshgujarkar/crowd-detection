def detect_crowd(person_count, location="general"):

    thresholds = {
        "mall": 20,
        "campus": 15,
        "street": 25,
        "general": 20
    }

    threshold = thresholds.get(location, 20)

    if person_count > threshold:
        return True, "CROWD RISK"

    return False, "Normal Crowd"