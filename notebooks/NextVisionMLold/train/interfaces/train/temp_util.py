from alibi.confidence import TrustScore
import pandas as pd
def get_trust_scores(train_X, train_y, test_X, test_pred):  
    #test_pred = pd.DataFrame(test_pred)  
    #class_mapping = {"low": 0, "low-med": 1, "medium": 2, "med-high": 3, "high": 4}
    #train_y= train_y["Risk Level"].map(class_mapping)
    #test_pred = test_pred[0].map(class_mapping)
    ts = TrustScore()
    ts.fit(train_X, train_y, classes=5)
    score, closest_class = ts.score(test_X, test_pred, k=2)
    return score