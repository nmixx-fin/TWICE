from __future__ import annotations

from finance_mteb.abstasks.TaskMetadata import TaskMetadata

from ....abstasks import AbsTaskClassification
from ....eval_instruction import kor_task2instruction

class KorFOMCClassification(AbsTaskClassification):
    metadata = TaskMetadata(
        name="KorFOMCClassification",
        description=kor_task2instruction["KorFOMCClassification"],
        reference="Financial News (Naver News)",
        dataset={
            "path": "nmixx-fin/twice_kr_fomc_cls",
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