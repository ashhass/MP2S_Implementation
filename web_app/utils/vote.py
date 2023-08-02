'''
    1. Generate prediction from supervised model
    2. Generate prediction from unsupervised model
    3. Take both votes to make final anomaly/not anomaly decision
'''

def majority_voting(loss, cls_probability):
    vote = 0
    # model 1 vote
    if loss > 0.0002:
        vote += 1

    # model 2 vote
    if cls_probability[0][0] > 0.21 and cls_probability[0][1] < 0.5:
        vote += 1
    print(loss, cls_probability[0][0], cls_probability[0][1])
    return True if vote == 2 else False 