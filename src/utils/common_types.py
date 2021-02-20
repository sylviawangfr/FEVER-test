import config


class PipelineParas(object):
    original_data = None
    upstream_data = None
    output_folder = None
    log_foder = None
    BERT_model = None
    BERT_tokenizer = None
    pred=False
    mode='eval'
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
        return config.RESULT_PATH / f"{self.output_folder}/{task}.jsonl"

    def get_eval_item_file(self, task):
        return config.RESULT_PATH / f"{self.output_folder}/{task}.jsonl"
