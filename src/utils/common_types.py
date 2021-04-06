import config


class PipelineParas(object):
    original_data = None
    upstream_data = None
    output_folder = None
    log_foder = None
    BERT_model = None
    BERT_tokenizer = None
    data_from_pred=False
    mode='eval' # mode in ['train', 'pred', 'eval']
    top_n = [5]
    prob_thresholds = 0.5
    sample_n = 5
    post_filter_prob = 1
    bert_client = None
    sampler = None

    def get_f1_log_file(self, task):
        return self.output_folder / f"bert_log/f1_analyze_{task}_{self.mode}.log"

    def get_eval_log_file(self, task):
        return self.output_folder / f"bert_log/eval_{task}_{self.mode}.log"

    def get_model_folder(self, task):
        return config.PRO_ROOT / f"saved_models/bert_finetuning/{task}_{self.mode}"

    def get_eval_result_file(self, task):
        return self.output_folder / f"{task}.jsonl"

    def get_eval_item_file(self, task):
        return self.output_folder / f"bert_log/item_{task}.jsonl"
