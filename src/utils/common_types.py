import config


class BERT_para(object):
    original_data = None
    upstream_data = None
    output_folder = None
    log_foder = None
    BERT_model = None
    BERT_tokenizer = None
    pred=False
    mode='dev'
    top_n = [5]
    prob_thresholds = 0.5
    sample_n = 5
    post_filter_prob = 1
    bert_client = None

    def get_f1_log_file(self, task):
        return config.LOG_PATH / f"{self.output_folder}/f1_analyze_{task}_{self.mode}_{self.prob_thresholds}_top{self.top_n}.log"

    def get_eval_log_file(self, task):
        return config.LOG_PATH / f"{self.output_folder}/eval_{task}_{self.mode}_{self.prob_thresholds}_top{self.top_n}.log"

    def get_model_folder(self, task):
        return config.PRO_ROOT / f"saved_models/bert_finetuning/{self.output_folder}/{task}_{self.mode}"

    def get_eval_data_file(self, task):
        return config.RESULT_PATH / f"{self.output_folder}/eval_data_{task}_{self.mode}_{self.prob_thresholds}_top{self.top_n}.jsonl"

    def get_eval_item_file(self, task):
        return config.RESULT_PATH / f"{self.output_folder}/eval_items_{task}_{self.mode}_{self.prob_thresholds}_top{self.top_n}.jsonl"

    def get_pred_data_file(self, task):
        return config.RESULT_PATH / f"{self.output_folder}/pred_data_{task}_{self.mode}_{self.prob_thresholds}_top{self.top_n}.jsonl"

    def get_pred_item_file(self, task):
        return config.RESULT_PATH / f"{self.output_folder}/pred_items_{task}_{self.mode}_{self.prob_thresholds}_top{self.top_n}.jsonl"