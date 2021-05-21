# Taken from https://github.com/asu-cactus/TransferHub/tree/master/transferhub

import numpy as np

def f_score(precision, recall, beta=1):
    """Compute F beta score
    
    Args:
        precision (float): precision
        recall (float): recall
        beta (float): the weight of recall, default=1
    
    Returns:
        float: f score
    """
    
    if recall + precision*beta**2 == 0:
        return 0
    else:        
        return (1 + beta**2)*precision*recall / (recall + precision*beta**2)

def accuracy_metrics(act, exp):
    """Return precision and recall for a single query
        
    Args:
        act (set): actual result
        exp (set): expected result
    
    Returns:
        float, float: precision and recall
    """
    
    if act == exp:
        # Consider the edge case: act=[] exp=[]
        return [1, 1]

    precision = len(act.intersection(exp)) / len(act) if len(act) != 0 else 0
    recall = len(act.intersection(exp)) / len(exp) if len(exp) != 0 else 0

    return precision, recall

def evaluation(act_dict, exp_dict):
    """Evaluate the accuracy of a actual result dict
    The dictionary organized in the following format:
        {'query_table_id': ['retrieved_table_id']}
        Example: {'q1', :['q2','q2','q3'], 'q2': ['q1', 'q5']}
    
    Args:
        act_dict (defaultdict(set)): actual result
        exp_dict (defaultdict(set)): expected result
    
    Returns:
        array with size(1,4): [precision, recall, f1, f05]
    """

    size = len(act_dict)
    # Construct a nx4 array to store accuracy metric values, n is the number of 
    # results
    # col0: precision, col1: recall
    # col2: f1 score,  col3: f0.5 score
    performance_matrics = np.zeros((size, 4))
    
    index = 0

    if len(act_dict.keys()) == 0:
        # Edge case: act_dict={}
        return np.array([0, 0, 0, 0])

    for key in act_dict.keys():
        act = act_dict[key]
        exp = exp_dict[key]
        precision, recall = accuracy_metrics(set(act), set(exp))
        f1 = f_score(precision, recall)
        f05 = f_score(precision, recall, beta=0.5)
        
        performance_matrics[index, :] = (precision, recall, f1, f05)
        index += 1
    
    return np.average(performance_matrics, axis=0)