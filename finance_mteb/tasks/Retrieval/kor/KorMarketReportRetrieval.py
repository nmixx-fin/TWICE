from __future__ import annotations

from finance_mteb.abstasks.TaskMetadata import TaskMetadata

from ....abstasks import AbsTaskRetrieval

class KorMarketReportRetrieval(AbsTaskRetrieval):
    metadata = TaskMetadata(
        name="KorMarketReportRetrieval",
        description="주식 시장과 관련된 금융 텍스트에 대해 답변에 도움이 될 문서를 검색합니다.",
        reference="Intenal Dataset",
        dataset={
            "path": "nmixx-fin/twice_kr_market_report_retrieval",
            "revision": "main",
        },
        type="Retrieval",
        category="s2p",
        eval_splits=["train"],
        eval_langs=["kor-Hang"],
        main_score="ndcg_at_10",
    )