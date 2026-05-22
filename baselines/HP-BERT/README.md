<h2><a href="https://link.springer.com/article/10.1007/s10660-022-09560-w">HP-BERT - Helpfulness Prediction BERT</a></h2>

사전학습 BERT의 [CLS] 표현을 frozen 상태로 활용해 리뷰 유용성을 회귀하는 단일 텍스트 모달 베이스라인

<hr>

<h2>🏛️Architecture</h2>
<p align="center">
  <img src="HP-BERT_picture.png" width="800" />
</p>
<p align="center">
  <i>Bilal, M., & Almazroi, A. A. (2023). Effectiveness of fine-tuned BERT model in classification of helpful and unhelpful online customer reviews. Electronic Commerce Research, 23, 2737-2757.</i>
</p>

<hr>

<h2>🔎Overview</h2>

- BERT 기반 모델 (사전학습 언어모델을 통한 텍스트 의미 표현 파악)
- 본 구현: Steam 리뷰 데이터에 동일 구조 적용(frozen), 사전학습 의미 표현을 회귀 task에 활용.

<hr>

<h2>Using tool</h2>

- Python 3.10-3.12, RTX 4090 / CUDA 12.x (RunPod)
- <code>torch==2.5.1</code> (+cu121) - BERT 추론 (frozen, no_grad)
- <code>transformers>=4.40.0</code>, <code>tokenizers>=0.15.0</code> - bert-base-uncased
- <code>tensorflow==2.18.1</code> - Keras 회귀 헤드, EarlyStopping, Adam
- <code>scikit-learn>=1.3</code> - split, MAE/MSE/MAPE 메트릭
- <code>numpy&lt;2.0</code>, <code>pandas>=2.0</code>, <code>tqdm</code>

<hr>

<h2>🎯Target</h2>

   * 학습 타깃:
      원시 votes_up 안정 회귀 위해 로그 변환 적용 → <code>log(votes_up + 1)</code>
   * q99 cutoff:
      long-tail 영향 완화 위해 train split의 99 퍼센타일 초과 샘플 제거
   * 데이터 누수 방지:
      q99 cutoff는 train split에만 적용, validation/test 미적용

<hr>


<h2>📝Architecture Details</h2>

  <h3>1) Embedding Single Text Modality</h3>

  * 입력 데이터:
      단일 텍스트 모달리티(리뷰 텍스트) 입력
  * 토큰화 및 임베딩:
      - 토큰화:
        bert-base-uncased tokenizer
      - 시퀀스 길이:
        최대 256 토큰 (padding / truncation)
      - 12층 Transformer Encoder, bert-base-uncased    
        - 가중치 frozen(본 프로젝트와 Yelp 데이터의 도메인 차이를 고려하여 Yelp fine-tuned 가중치를 사용하지 않음)

  <h3>2) [CLS] 표현 추출</h3>

  - self-attention 기반 양방향 문맥 정보를 압축한 [CLS] 토큰의 <code>pooler_output</code>
  - 마지막 hidden state에 tanh 활성을 거친 표현
  - 768차원 벡터로 한 리뷰의 통합 표현 확보

<hr>

<h2>Regression Head</h2>

* Dropout → Dense(1, linear)

<hr>
