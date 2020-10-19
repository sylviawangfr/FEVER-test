from utils.file_loader import *

def create_submmission(input_data):
    new_result = []
    for i in input_data:
        one_new_item = {'id': i['id'], 'predicted_label': i['predicted_label'], 'predicted_evidence': i['predicted_evidence']}
        new_result.append(one_new_item)

    orginal_data = read_json_rows(config.FEVER_TEST_JSONL)
    assert len(orginal_data) == len(new_result)
    for i in range(len(orginal_data)):
        assert orginal_data[i]['id'] == new_result[i]['id']

    save_intermidiate_results(new_result, config.RESULT_PATH / 'predictions_gat.jsonl')


if __name__ == '__main__':
    # input_data = read_json_rows(config.RESULT_PATH / 'nli_test_pred_full/predictions_org.jsonl')
    input_data = read_json_rows(config.RESULT_PATH / 'nli_test_bert_gat/eval_data_nli_test_0.5_top[5].jsonl')
    create_submmission(input_data)