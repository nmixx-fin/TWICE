from __future__ import annotations

from finance_mteb.abstasks.TaskMetadata import TaskMetadata

from ....abstasks import AbsTaskRetrieval

class KorTATQARetrieval(AbsTaskRetrieval):
    metadata = TaskMetadata(
        name="KorTATQARetrieval",
        description="재무 리포트에서 주어진 금융 텍스트에 대해 가장 적절한 사용자 답변을 검색합니다.",
        reference="Intenal Dataset",
        dataset={
            "path": "nmixx-fin/twice_tat_qa_retrieval",
            "revision": "main",
        },
        type="Retrieval",
        category="s2p",
        eval_splits=["train"],
        eval_langs=["kor-Hang"],
        main_score="ndcg_at_10",
    )