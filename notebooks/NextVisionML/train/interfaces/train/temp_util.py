from alibi.confidence import TrustScore

def get_trust_scores(X, y):
    ts = TrustScore()
    #classes = {"low", "low-med", "medium", "med-high", "high"}
    score, closest_class = ts.score(X, y, k=4)
    return score