from sklearn.metrics import f1_score
from utils import c_scorer


def simple_accuracy(preds, labels):
    return (preds == labels).mean()


def acc_and_f1(preds, labels, average='binary'):
    acc = simple_accuracy(preds, labels)
    f1 = f1_score(y_true=labels, y_pred=preds, average=average)
    return {
        "total": len(preds),
        "correct_hit": (preds==labels).tolist().count(True),
        "acc": acc,
        "f1": f1,
        "acc_and_f1": (acc + f1) / 2,
    }


def compute_metrics(preds, labels, average='binary'):
    assert len(preds) == len(labels)
    return {"acc": acc_and_f1(preds, labels, average)}


def convert_evidence2scoring_format(predicted_sentids):
    e_list = predicted_sentids
    pred_evidence_list = []
    for i, cur_e in enumerate(e_list):
        doc_id = cur_e.split(c_scorer.SENT_LINE)[0]
        ln = cur_e.split(c_scorer.SENT_LINE)[1]
        pred_evidence_list.append([doc_id, int(ln)])
    return pred_evidence_list