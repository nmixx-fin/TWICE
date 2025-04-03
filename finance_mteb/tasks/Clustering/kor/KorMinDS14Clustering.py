from __future__ import annotations

from finance_mteb.abstasks.AbsTaskClustering import AbsTaskClustering
from finance_mteb.abstasks.AbsTaskClusteringFast import clustering_downsample
from finance_mteb.abstasks.TaskMetadata import TaskMetadata
import ast
import pandas as pd
import json
from datasets import Dataset, DatasetDict

class KorMInDS14Clustering(AbsTaskClustering):
    metadata = TaskMetadata(
        name="KorMInDS14Clustering",
        description="사업 보고서에 나와있는 기업 설명 텍스트를 기반으로 산업군을 분류합니다.",
        reference="https://arxiv.org/pdf/2104.08524",
        dataset={
            "path": "nmixx-fin/twice_kr_minds14_clustering",
            "revision": "main",
        },
        type="Clustering",
        category="p2p",
        eval_splits=["test"],
        eval_langs=["kor-Hang"],
        main_score="v_measure",
    )

    def dataset_transform(self):
        import pandas as pd
        import json
        from datasets import Dataset, DatasetDict
        
        # 데이터셋을 DataFrame으로 변환
        self.dataset = pd.DataFrame(self.dataset['test'])
        
        # 데이터 형식 확인 및 디버깅
        print("Dataset columns:", self.dataset.columns)
        print("Sample sentences type:", type(self.dataset['sentences'].iloc[0]))
        print("Sample sentences value:", self.dataset['sentences'].iloc[0])
        
        # 문자열로 저장된 리스트를 실제 리스트로 변환
        try:
            # 문자열이 JSON 형식인 경우
            self.dataset['sentences'] = self.dataset['sentences'].apply(lambda x: json.loads(x) if isinstance(x, str) else x)
            self.dataset['labels'] = self.dataset['labels'].apply(lambda x: json.loads(x) if isinstance(x, str) else x)
        except Exception as e:
            print(f"JSON parsing error: {e}")
            try:
                # 문자열이 Python 리스트 형식인 경우
                import ast
                self.dataset['sentences'] = self.dataset['sentences'].apply(lambda x: ast.literal_eval(x) if isinstance(x, str) else x)
                self.dataset['labels'] = self.dataset['labels'].apply(lambda x: ast.literal_eval(x) if isinstance(x, str) else x)
            except Exception as e:
                print(f"AST parsing error: {e}")
                # 파싱이 실패하면 문자열을 직접 처리
                self.dataset['sentences'] = self.dataset['sentences'].apply(lambda x: [x] if isinstance(x, str) else x)
                self.dataset['labels'] = self.dataset['labels'].apply(lambda x: [x] if isinstance(x, str) else x)
        
        # 리스트 내부의 모든 항목이 문자열인지 확인
        self.dataset['sentences'] = self.dataset['sentences'].apply(lambda x: [str(item) for item in x] if isinstance(x, list) else [str(x)])
        self.dataset['labels'] = self.dataset['labels'].apply(lambda x: [str(item) for item in x] if isinstance(x, list) else [str(x)])
        
        # DataFrame을 Hugging Face Dataset으로 변환
        dataset_hf = Dataset.from_pandas(self.dataset)
        
        # test split만 포함한 DatasetDict 형태로 반환
        self.dataset = DatasetDict({"test": dataset_hf})