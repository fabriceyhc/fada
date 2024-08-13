import torch
import numpy as np
import evaluate
from transformers import pipeline, AutoTokenizer
from huggingface_hub import HfApi

from fada.utils import vectorize, ConfiguredMetric

class PerformanceExtractor:
    def __init__(self, 
                 builder_name=None, 
                 config_name=None, 
                 model=None, 
                 tokenizer=None,
                 model_id=None):
        self.builder_name = builder_name
        self.config_name = config_name
        self.model = model
        self.tokenizer = tokenizer
        self.model_id = model_id
        self.api = HfApi()
        self.device = 0 if torch.cuda.is_available() else -1
        self.pipe = None

        # initializations
        self.find_model_for_dataset()
        self.metrics = evaluate.combine([
                evaluate.load('accuracy'), 
                ConfiguredMetric(evaluate.load('precision'), average='weighted'),
                ConfiguredMetric(evaluate.load('recall'), average='weighted'),
                ConfiguredMetric(evaluate.load('f1'), average='weighted'),
            ])

    def compute_metrics(self, preds, labels):
        return self.metrics.compute(
            predictions=np.array(preds), 
            references=np.array(labels))

    def find_model_for_dataset(self):
        if self.model is None:
            if not self.model_id:
                # search for appropriate model based on dataset 
                if self.config_name in ["default", "plain_text"]:
                    search_name = self.builder_name
                else:
                    search_name = self.config_name

                model_id = next(iter(self.api.list_models(task="text-classification", library="pytorch", trained_dataset=search_name)))
                model_id = getattr(model_id, 'id')
            else:
                # use provided model_id string
                model_id = self.model_id
            print(f"Using {model_id} to measure performance.")
            self.pipe = pipeline(
                "text-classification",
                model=model_id,
                tokenizer=model_id,
                device=self.device,
                max_length=256,
                truncation=True,
                top_k=None
            )
        else:
            self.pipe = pipeline(
                "text-classification", 
                model=self.model, 
                tokenizer=self.tokenizer,
                device=self.device,
                max_length=256,
                truncation=True,
                top_k=None)


    def extract_prediction_probabilities(self, dataset):
        output = self.pipe(dataset['text'])
        return np.stack([vectorize(o) for o in output])

    def annotate_preds(self, dataset):
        if "probs" in dataset.features:
            dataset = dataset.remove_columns(["probs"])
        if "preds" in dataset.features:
            dataset = dataset.remove_columns(["preds"])
        probs = self.extract_prediction_probabilities(dataset)
        preds = np.argmax(probs, axis=1)
        dataset = dataset.add_column("probs", [p for p in probs])
        dataset = dataset.add_column("preds", [p for p in preds])
        return dataset, np.array(preds)

    def extract_performance(self, dataset, force=True):
        if "preds" not in dataset.features or force:
            dataset, preds = self.annotate_preds(dataset)
        return self.compute_metrics(preds, dataset["label"])

if __name__ == "__main__":
    from datasets import load_dataset

    builder_name, config_name = "ag_news", "default"
    dataset = load_dataset(builder_name, config_name, split="test")
    pe = PerformanceExtractor(model_id="textattack/bert-base-uncased-ag-news")
    perf = pe.extract_performance(dataset)
    print(perf)
