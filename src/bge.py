import pandas as pd
import numpy as np
from tqdm import tqdm
from typing import List, Dict, Optional

import torch
from transformers import AutoTokenizer, AutoModel

class BGEExtractor:
    def __init__(
        self,
        model_ckpt: str = "BAAI/bge-large-en-v1.5",
        batch_size: int = 128,
        chunk_size: int = 510,
        max_length: int = 512,
        emb_dim: int = 1024,
        output_col_review: str = "review_bge",
        output_col_patch: str = "patch_bge",
        verbose: bool = True,
    ):
        self.model_ckpt = model_ckpt
        self.batch_size = batch_size
        self.chunk_size = chunk_size
        self.max_length = max_length
        self.emb_dim = emb_dim
        self.output_col_review = output_col_review
        self.output_col_patch = output_col_patch
        self.verbose = verbose

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.tokenizer, self.model = self._model()

    def _model(self):
        tokenizer = AutoTokenizer.from_pretrained(self.model_ckpt)
        model = AutoModel.from_pretrained(self.model_ckpt).to(self.device)
        model.eval()
        return tokenizer, model

    @torch.inference_mode()
    def _encode_reviews(self, texts: List[str], tag: str = "reviews") -> np.ndarray:
        N = len(texts)
        cls_arr = np.zeros((N, self.emb_dim), dtype=np.float32)
        pbar = tqdm(total=N, desc=f"encode {tag}", unit="rows") if self.verbose else None

        for start in range(0, N, self.batch_size):
            batch = texts[start:start + self.batch_size]
            enc = self.tokenizer(
                batch, padding=True, truncation=True,
                max_length=self.max_length, return_tensors="pt",
            )
            ids  = enc["input_ids"].to(self.device)
            amsk = enc["attention_mask"].to(self.device)

            hidden = self.model(input_ids=ids, attention_mask=amsk).last_hidden_state
            cls_vec = hidden[:, 0, :]
            cls_arr[start:start + cls_vec.size(0)] = cls_vec.cpu().float().numpy()

            if pbar is not None:
                pbar.update(len(batch))

            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        if pbar is not None:
            pbar.close()
        return cls_arr

    @torch.inference_mode()
    def _encode_patch_text(self, text: str) -> np.ndarray:
        token_ids = self.tokenizer.encode(text, add_special_tokens=False)
        chunks = []
        for j in range(0, len(token_ids), self.chunk_size):
            chunk = token_ids[j:j + self.chunk_size]
            chunks.append(self.tokenizer.decode(chunk, skip_special_tokens=True))
        if not chunks:
            chunks = [text[:200]]

        cls_list = []
        for ci in range(0, len(chunks), self.batch_size):
            batch_chunks = chunks[ci:ci + self.batch_size]
            enc = self.tokenizer(
                batch_chunks, padding=True, truncation=True,
                max_length=self.max_length, return_tensors="pt",
            )
            ids  = enc["input_ids"].to(self.device)
            amsk = enc["attention_mask"].to(self.device)
            h = self.model(input_ids=ids, attention_mask=amsk).last_hidden_state
            cls_list.append(h[:, 0, :])

        all_cls = torch.cat(cls_list, dim=0)
        return all_cls.mean(dim=0).cpu().float().numpy()

    def run(self, df: pd.DataFrame, text_col: str, output_col: Optional[str] = None) -> pd.DataFrame:
        if text_col not in df.columns:
            raise KeyError(f"{text_col} column not found in DataFrame.")

        output_col = output_col or self.output_col_review
        df = df.copy()
        df[text_col] = df[text_col].fillna("").astype(str)

        embs = self._encode_reviews(df[text_col].tolist(), tag=output_col)
        df[output_col] = embs.tolist()
        return df

    def run_patches(self, patch_dict: Dict[str, str]) -> Dict[str, np.ndarray]:
        names = sorted(patch_dict.keys())
        out = {}
        for name in tqdm(names, desc="encode patches") if self.verbose else names:
            out[name] = self._encode_patch_text(patch_dict[name])
        return out