#필요 라이브러리
import os
import re
import pandas as pd
import numpy as np
import contractions
import textstat
from textblob import TextBlob
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from tqdm import tqdm
tqdm.pandas()
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

#데이터 저장
from src.path import PROCESSED_PATH, RAW_PATH, PATCH_DIR

#BGE 임베딩
from src.bge import BGEExtractor

HC_COLS = ['hc_word_count', 'hc_readability', 'hc_subjectivity', 'hc_sentiment']

class DataProcessor:
    def __init__(self, fname: str,
                 text_col: str=None, rating_col: str=None,
                 patch_name_col: str=None, patch_flag_col: str=None,
                 meta_num_cols: list=None, meta_bin_cols: list=None,
                 min_words: int=3, clip_quantile: float=0.99,
                 test_val_size: float=None, val_size: float=None,
                 bge_config: dict=None):
        self.fname = fname

        #데이터 로드
        self.text_col = text_col
        self.rating_col = rating_col
        self.patch_name_col = patch_name_col
        self.patch_flag_col = patch_flag_col
        self.meta_num_cols = meta_num_cols or []
        self.meta_bin_cols = meta_bin_cols or []
        self.meta_cols = self.meta_num_cols + self.meta_bin_cols
        self.min_words = min_words

        # 데이터 분할
        self.clip_quantile = clip_quantile
        self.test_val_size = test_val_size
        self.val_size = val_size

        # BGE
        self.bge_config = bge_config or {}

        # VADER
        self._vader = SentimentIntensityAnalyzer()

    #텍스트 정제
    def _clean_text(self, text):
        text = str(text)
        text = re.sub(r"https?://\S+|www\.\S+", " ", text)
        text = re.sub(r"<[^>]+>", " ", text)
        try:
            text = contractions.fix(text)
        except IndexError:
            return None
        text = re.sub(r"[^A-Za-z0-9\s]", " ", text)
        text = text.lower()
        text = re.sub(r"\s+", " ", text).strip()
        if len(text.split()) < self.min_words:
            return None
        return text

    #데이터 로드
    def load_data(self, dpath: str):
        print("Loading Data...")
        usecols = [self.text_col, self.rating_col,
                   self.patch_name_col, self.patch_flag_col] + self.meta_cols
        df = pd.read_csv(dpath, usecols=usecols,
                         on_bad_lines='skip', engine='python')
        print(f'원본 로드: {len(df):,}건')

        # 이진형 메타 NaN → 0
        for c in self.meta_bin_cols:
            df[c] = df[c].fillna(0).astype(int)

        # 패치 매칭 플래그 필터
        n_before = len(df)
        df = df[df[self.patch_flag_col] == True].reset_index(drop=True)
        print(f'[필터] 패치 매칭 없음 → {n_before - len(df):,}건 제거')

        # NaN drop
        n_before = len(df)
        df = df.dropna(subset=[self.text_col, self.rating_col,
                               self.patch_name_col] + self.meta_num_cols).reset_index(drop=True)
        n_dropped = n_before - len(df)
        if n_dropped > 0:
            print(f'[필터] NaN → {n_dropped:,}건 제거')

        df[self.text_col] = df[self.text_col].astype(str)
        df[self.patch_name_col] = df[self.patch_name_col].astype(str)

        # 텍스트 정제
        df['review_clean'] = df[self.text_col].apply(self._clean_text)
        n_before = len(df)
        df = df[df['review_clean'].notna()].reset_index(drop=True)
        n_dropped = n_before - len(df)
        if n_dropped > 0:
            print(f'[필터] 클린텍스트 무효 (None) → {n_dropped:,}건 제거')

        # rating >= 1
        n_before = len(df)
        df = df[df[self.rating_col] >= 1].reset_index(drop=True)
        print(f'[필터] {self.rating_col} < 1 → {n_before - len(df):,}건 제거')

        df = df.rename(columns={self.rating_col: "rating"})
        print("Data Loaded!")
        return df

    #패치노트 로드 + 매칭 필터
    def load_patches(self, df: pd.DataFrame):
        patch_texts = {}
        for p in sorted(os.listdir(PATCH_DIR)):
            if not p.endswith('.txt'):
                continue
            fpath = os.path.join(PATCH_DIR, p)
            try:
                with open(fpath, 'r', encoding='utf-8', errors='replace') as f:
                    patch_texts[p] = f.read()
            except OSError:
                pass

        assert len(patch_texts) > 0, f'패치노트가 없습니다: {PATCH_DIR}'

        no_patch_mask = ~df[self.patch_name_col].isin(patch_texts)
        if no_patch_mask.sum() > 0:
            print(f'[필터] 패치 파일 불일치 → {no_patch_mask.sum():,}건 제거')
        df = df[~no_patch_mask].reset_index(drop=True)

        assert len(df) > 0, '필터링 후 데이터 0건'
        print(f'\n최종 데이터: {len(df):,}건')
        print(f'패치 파일: {len(patch_texts)}개  |  사용된 패치: {df[self.patch_name_col].nunique()}개')
        return df, patch_texts

    #Hand-crafted 피처 추출
    def extract_handcrafted(self, df: pd.DataFrame):
        def _extract(text: str) -> dict:
            t = text if isinstance(text, str) else ''
            words = re.findall(r'\w+', t)
            n_word = len(words)
            try:
                flesch = textstat.flesch_reading_ease(t) if n_word > 3 else 0.0
            except Exception:
                flesch = 0.0
            try:
                subjectivity = TextBlob(t).sentiment.subjectivity
            except Exception:
                subjectivity = 0.0
            compound = self._vader.polarity_scores(t)['compound']
            return {
                'hc_word_count': float(n_word),
                'hc_readability': flesch,
                'hc_subjectivity': subjectivity,
                'hc_sentiment': compound,
            }

        rows = [_extract(t) for t in tqdm(df['review_clean'].tolist(), desc='handcrafted')]
        hc_df = pd.DataFrame(rows)
        df = pd.concat([df.reset_index(drop=True), hc_df.reset_index(drop=True)], axis=1)
        print('HC features:', HC_COLS, '| shape:', hc_df.shape)
        return df

    # 데이터 분할 + 스케일링 + 저장
    def split_data(self, df: pd.DataFrame):
        test_size = (self.test_val_size - self.val_size) / self.test_val_size
        train, val_test = train_test_split(df, test_size=self.test_val_size, random_state=42)
        val, test = train_test_split(val_test, test_size=test_size, random_state=42)
        train = train.reset_index(drop=True)
        val   = val.reset_index(drop=True)
        test  = test.reset_index(drop=True)
        print(f"Train size: {len(train):,}, Val size: {len(val):,}, Test size: {len(test):,}")

        # Q99 클리핑 (train fit → 전체 적용)
        Q = train['rating'].quantile(self.clip_quantile)
        print(f'[클리핑] rating Q{int(self.clip_quantile*100)} (train 기준) = {Q:.0f}')
        for d in (train, val, test):
            d['rating'] = d['rating'].clip(upper=Q)

        # y_log = log1p(rating)
        for d in (train, val, test):
            d['y_log'] = np.log1p(d['rating'].astype(float))

        # 수치형 메타: log1p
        for d in (train, val, test):
            for c in self.meta_num_cols:
                d[c] = np.log1p(d[c].clip(lower=0).astype(float))

        # 수치형 메타: StandardScaler (train fit)
        meta_scaler = StandardScaler().fit(train[self.meta_num_cols].values.astype(float))
        for d in (train, val, test):
            d[self.meta_num_cols] = meta_scaler.transform(
                d[self.meta_num_cols].values.astype(float)
            ).astype(np.float32)

        # HC: StandardScaler (train fit)
        hc_scaler = StandardScaler().fit(train[HC_COLS].values)
        for d in (train, val, test):
            d[HC_COLS] = hc_scaler.transform(d[HC_COLS].values).astype(np.float32)

        # 저장 컬럼만 유지
        keep = ['review_bge', 'patch_bge'] + HC_COLS + self.meta_cols + ['y_log']
        train = train[keep]
        val   = val[keep]
        test  = test[keep]

        train.to_parquet(os.path.join(PROCESSED_PATH, f"{self.fname}_train.parquet"))
        val.to_parquet(os.path.join(PROCESSED_PATH, f"{self.fname}_val.parquet"))
        test.to_parquet(os.path.join(PROCESSED_PATH, f"{self.fname}_test.parquet"))

    #Processing RUN
    def run(self):
        dpath = os.path.join(RAW_PATH, f"{self.fname}.csv")
        df = self.load_data(dpath)
        df, patch_texts = self.load_patches(df)
        df = self.extract_handcrafted(df)

        extractor = BGEExtractor(**self.bge_config)

        # 리뷰 BGE
        df = extractor.run(df, text_col='review_clean',
                           output_col=extractor.output_col_review)

        # 패치 BGE → 각 리뷰 행에 매칭
        patch_bge_dict = extractor.run_patches(patch_texts)
        df[extractor.output_col_patch] = df[self.patch_name_col].map(
            lambda n: patch_bge_dict[n].tolist()
        )

        self.split_data(df)
