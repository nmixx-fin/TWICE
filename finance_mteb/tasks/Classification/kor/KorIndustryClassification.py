from __future__ import annotations

from finance_mteb.abstasks.TaskMetadata import TaskMetadata

from ....abstasks import AbsTaskClassification
from ....eval_instruction import kor_task2instruction

class KorIndustryClassification(AbsTaskClassification):
    metadata = TaskMetadata(
        name="KorIndustryClassification",
        description=kor_task2instruction["KorIndustryClassification"],
        reference="Industry/Investment Analysis Report",
        dataset={
            "path": "nmixx-fin/twice_kr_industry_cls",
            "revision": "main",
        },
        type="Classification",
        category="s2s",
        eval_splits=["train"],
        eval_langs=["kor-Hang"],
        main_score="accuracy",
    )
    def dataset_transform(self):
        self.dataset = self.dataset.rename_column("sentence", "text")