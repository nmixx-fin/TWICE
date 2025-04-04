from __future__ import annotations

from finance_mteb.abstasks.TaskMetadata import TaskMetadata

from ....abstasks import AbsTaskRetrieval

class KorFSSDictRetrieval(AbsTaskRetrieval):
    metadata = TaskMetadata(
        name="KorFSSDictRetrieval",
        description="주어진 금융 용어에 대해 답변에 도움이 될 문서를 검색합니다.",
        reference="Intenal Dataset",
        dataset={
            "path": "nmixx-fin/twice_fss_dict_retrieval",
            "revision": "main",
        },
        type="Retrieval",
        category="s2p",
        eval_splits=["train"],
        eval_langs=["kor-Hang"],
        main_score="ndcg_at_10",
    )