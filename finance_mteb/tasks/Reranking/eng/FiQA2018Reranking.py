from __future__ import annotations

from finance_mteb.abstasks.TaskMetadata import TaskMetadata

from ....abstasks.AbsTaskReranking import AbsTaskReranking


class FiQA2018Reranking(AbsTaskReranking):
    metadata = TaskMetadata(
        name="FiQA2018Reranking",
        description="Financial opinion mining and question answering",
        reference="https://sites.google.com/view/fiqa/",
        dataset={
            "path": "nmixx-fin/twice_ko-trans_fiqa2018_reranking",
            "revision": "main",
        },
        type="Reranking",
        category="s2s",
        eval_splits=["test"],
        eval_langs=["kor-Hang"],
        main_score="map",
    )

