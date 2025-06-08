from __future__ import annotations

import logging
import json
import os
from pathlib import Path
from time import time

from finance_mteb.abstasks.TaskMetadata import TaskMetadata

from ....abstasks import AbsTaskRetrieval

logger = logging.getLogger(__name__)

class KorBoKDictRetrieval(AbsTaskRetrieval):
    metadata = TaskMetadata(
        name="KorBoKDictRetrieval",
        description="주어진 금융 용어에 대해 답변에 도움이 될 문서를 검색합니다.",
        reference="", # reference 추가
        dataset={
            "path": "nmixx-fin/twice_bok_dict_retrieval",
            "revision": "main",
        },
        type="Retrieval",
        category="s2p",
        eval_splits=["train"],
        eval_langs=["kor-Hang"],
        main_score="ndcg_at_10",
    )