from __future__ import annotations

import logging
from collections import defaultdict
from typing import Any

import numpy as np

from ..evaluation.evaluators import (
    kNNClassificationEvaluator,
    kNNClassificationEvaluatorPytorch,
    logRegClassificationEvaluator,
)
from ..MTEBResults import HFSubset, ScoresDict
from .AbsTask import AbsTask

logger = logging.getLogger(__name__)


class AbsTaskClassification(AbsTask):
    """Abstract class for kNN classification tasks
    The similarity is computed between pairs and the results are ranked.

    self.load_data() must generate a huggingface dataset with a split matching self.metadata_dict["eval_splits"], and assign it to self.dataset. It
    must contain the following columns:
        text: str
        label: int
    """

    def __init__(
        self,
        method: str = "logReg",
        n_experiments: int | None = None,
        samples_per_label: int | None = None,
        k: int = 3,
        batch_size: int = 32,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.batch_size = batch_size
        self.method = method

        # Bootstrap parameters
        self.n_experiments: int = (  # type: ignore
            n_experiments
            if n_experiments is not None
            else self.metadata_dict.get("n_experiments", 10)
        )
        self.samples_per_label: int = (  # type: ignore
            samples_per_label
            if samples_per_label is not None
            else self.metadata_dict.get("samples_per_label", 8)
        )

        # kNN parameters
        self.k = k

        # Run metadata validation by instantiating addressing the attribute
        # This is quite hacky. Ideally, this would be done in the constructor of
        # each concrete task, but then we have to duplicate the __init__ method's
        # interface.
        if hasattr(self, "metadata"):
            self.metadata

    def _add_main_score(self, scores: dict[HFSubset, ScoresDict]) -> None:
        scores["main_score"] = scores[self.metadata.main_score]

    def evaluate(
        self, model, eval_split="test", train_split="train", **kwargs
    ) -> dict[HFSubset, ScoresDict]:
        if not self.data_loaded:
            self.load_data()

        scores = {}
        hf_subsets = [l for l in self.dataset] if self.is_multilingual else ["default"]

        for hf_subset in hf_subsets:
            logger.info(
                f"\nTask: {self.metadata.name}, split: {eval_split}, subset: {hf_subset}. Running..."
            )

            if hf_subset not in self.dataset and hf_subset == "default":
                ds = self.dataset
            else:
                ds = self.dataset[hf_subset]
            scores[hf_subset] = self._evaluate_subset(
                model, ds, eval_split, train_split, **kwargs
            )
            self._add_main_score(scores[hf_subset])

        return scores

    def _evaluate_subset(
        self, model, dataset, eval_split="test", train_split="train", **kwargs
    ) -> ScoresDict:
        train_split = dataset[train_split]
        eval_split = dataset[eval_split]
        params = {"k": self.k, "batch_size": self.batch_size}
        params.update(kwargs)

        scores = []
        test_cache, idxs = (
            None,
            None,
        )  # we store idxs to make the shuffling reproducible
        for i in range(self.n_experiments):
            logger.info(
                "=" * 10 + f" Experiment {i+1}/{self.n_experiments} " + "=" * 10
            )
            # Bootstrap `self.samples_per_label` samples per label for each split
            X_sampled, y_sampled, idxs = self._undersample_data(
                train_split["text"], train_split["label"], self.samples_per_label, idxs
            )

            if "MCQA" in self.metadata.name:
                evaluator = self._get_mcqa_evaluator(
                    X_sampled, y_sampled, eval_split, **params
                )
            elif "BQA" in self.metadata.name:
                evaluator = self._get_bqa_evaluator(
                    X_sampled, y_sampled, eval_split, **params
                )
            elif "MMLU" in self.metadata.name:
                evaluator = self._get_mmlu_evaluator(
                    X_sampled, y_sampled, eval_split, **params
                )
            else:
                evaluator = self._get_standard_evaluator(
                    X_sampled, y_sampled, eval_split, **params
                )

            scores_exp, test_cache = evaluator(model, test_cache=test_cache)
            scores.append(scores_exp)


        avg_scores: dict[str, Any] = {
            k: np.mean([s[k] for s in scores]) for k in scores[0].keys()
        }
        avg_scores["scores_per_experiment"] = scores
        return avg_scores

    def _get_mcqa_evaluator(self, X_sampled, y_sampled, eval_split, **params):
        # "text"와 "query" 컬럼을 결합하여 평가를 위한 evaluator를 반환합니다.
        combined_input = [f"{t} {q}" for t, q in zip(eval_split["text"], eval_split["query"])]
        
        if self.method == "kNN":
            return kNNClassificationEvaluator(
                X_sampled,
                y_sampled,
                combined_input,
                eval_split["label"],
                eval_split["options"],
                task_name=self.metadata.name,
                **params,
            )
        elif self.method == "kNN-pytorch":
            return kNNClassificationEvaluatorPytorch(
                X_sampled,
                y_sampled,
                combined_input,
                eval_split["label"],
                eval_split["options"],
                task_name=self.metadata.name,
                **params,
            )
        elif self.method == "logReg":
            # return logRegClassificationEvaluator(
            #     X_sampled,
            #     y_sampled,
            #     combined_input,
            #     eval_split["label"],
            #     eval_split["options"],
            #     task_name=self.metadata.name,
            #     **params,
            # )

            # logRegClassificationEvaluator 생성자의 매개변수 순서를 정확히 맞추어 호출
            return logRegClassificationEvaluator(
                X_sampled,                  
                y_sampled,                  
                eval_split["text"],         
                eval_split["label"],        
                self.metadata.name,         # task_name (5번째 인자, 위치 인자로 전달)
                **params                    
            )
        else:
            raise ValueError(f"Method {self.method} not supported")
    
    def _get_bqa_evaluator(self, X_sampled, y_sampled, eval_split, **params):
        # "text"와 "query" 컬럼을 결합하여 평가를 위한 evaluator를 반환합니다.
        combined_input = [f"{t} {q}" for t, q in zip(eval_split["text"], eval_split["query"])]
        
        if self.method == "kNN":
            return kNNClassificationEvaluator(
                X_sampled,
                y_sampled,
                combined_input,
                eval_split["label"],
                task_name=self.metadata.name,
                **params,
            )
        elif self.method == "kNN-pytorch":
            return kNNClassificationEvaluatorPytorch(
                X_sampled,
                y_sampled,
                combined_input,
                eval_split["label"],
                task_name=self.metadata.name,
                **params,
            )
        elif self.method == "logReg":
            return logRegClassificationEvaluator(
                X_sampled,
                y_sampled,
                combined_input,
                eval_split["label"],
                task_name=self.metadata.name,
                **params,
            )
        else:
            raise ValueError(f"Method {self.method} not supported")

    def _get_mmlu_evaluator(self, X_sampled, y_sampled, eval_split, **params):
        # "query"와 "options" 컬럼을 사용하여 평가를 위한 evaluator를 반환합니다.
        
        # 수정 전
        # combined_input = [f"{q} Options: {o}" for q, o in zip(eval_split["query"], eval_split["options"])]
        
        # 수정 후
        combined_input = [f"{q} Options: {o}" for q, o in zip(eval_split["text"], eval_split["options"])]  # 'query' → 'text'

        if self.method == "kNN":
            return kNNClassificationEvaluator(
                X_sampled,
                y_sampled,
                combined_input,
                eval_split["label"],
                task_name=self.metadata.name,
                **params,
            )
        elif self.method == "kNN-pytorch":
            return kNNClassificationEvaluatorPytorch(
                X_sampled,
                y_sampled,
                combined_input,
                eval_split["label"],
                task_name=self.metadata.name,
                **params,
            )
        elif self.method == "logReg":
            return logRegClassificationEvaluator(
                X_sampled,
                y_sampled,
                combined_input,
                eval_split["label"],
                task_name=self.metadata.name,
                **params,
            )
        else:
            raise ValueError(f"Method {self.method} not supported")
        
    def _get_standard_evaluator(self, X_sampled, y_sampled, eval_split, **params):
        if self.method == "kNN":
            return kNNClassificationEvaluator(
                X_sampled,
                y_sampled,
                eval_split["text"],
                eval_split["label"],
                task_name=self.metadata.name,
                **params,
            )
        elif self.method == "kNN-pytorch":
            return kNNClassificationEvaluatorPytorch(
                X_sampled,
                y_sampled,
                eval_split["text"],
                eval_split["label"],
                task_name=self.metadata.name,
                **params,
            )
        elif self.method == "logReg":
            return logRegClassificationEvaluator(
                X_sampled,
                y_sampled,
                eval_split["text"],
                eval_split["label"],
                task_name=self.metadata.name,
                **params,
            )
        else:
            raise ValueError(f"Method {self.method} not supported")


    def _undersample_data(self, X, y, samples_per_label: int, idxs=None):
        """Undersample data to have samples_per_label samples of each label"""
        X_sampled = []
        y_sampled = []
        if idxs is None:
            idxs = np.arange(len(y))
        np.random.shuffle(idxs)
        label_counter = defaultdict(int)
        for i in idxs:
            if label_counter[y[i]] < samples_per_label:
                X_sampled.append(X[i])
                y_sampled.append(y[i])
                label_counter[y[i]] += 1
        return X_sampled, y_sampled, idxs
