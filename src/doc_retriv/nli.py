from tqdm import tqdm
import config
from BERT_test.nli_eval import nli_pred_evi_score_only
import utils.common_types as bert_para
from collections import Counter


def nli_vote(data_nli_with_score):
    id2label = {
        0: "SUPPORTS",
        1: "REFUTES",
        2: "NOT ENOUGH INFO"
    }
    with tqdm(total=len(data_nli_with_score), desc=f"searching triple sentences") as pbar:
        for idx, example in enumerate(data_nli_with_score):
            preds = example['evi_nli']
            pred_labels = [p['predicted_label'] for p in preds]
            count = Counter()
            count.update(pred_labels)
            label_count = sorted(list(count.most_common()), key=lambda x: x[0])


def nli_pred(data_nli, output_file):
    paras = bert_para.PipelineParas()
    paras.mode = 'eval'
    paras.upstream_data = data_nli
    paras.BERT_model = config.PRO_ROOT / "saved_models/bert_finetuning/nli_train_86.7"
    paras.BERT_tokenizer = config.PRO_ROOT / "saved_models/bert_finetuning/nli_train_86.7"
    paras.output_folder = output_file
    nli_pred_evi_score_only(paras)


if __name__ == '__main__':
    t = [1,1,1,1,2,1,0,1,2,0,0,0]
    count = Counter()
    count.update(t)
    print(count.most_common())
    print(sorted(list(count.most_common()), key=lambda x: x[0]))