<h1><a href="https://www.sciencedirect.com/science/article/pii/S0957417422002469">TNN - Text-based 1D Convolutional Neural Network</a></h1>

<hr>

<h2>🏛️Arcchitecture</h2>
<p align="center">
  <img src="TNN_picture.png" width="800" />
</p>
<p align="center">
  <i>Olmedilla, M., Martinez-Torres, M. R., & Toral, S. (2022). Prediction and modelling online reviews helpfulness using 1D Convolutional Neural Networks. Expert Systems with Applications, 198, 116787.</i>
</p>

<hr>

<h2>🔎Overview</h2>

- 전통적 1D CNN 기반 모델 (다양한 kernel size의 병렬 합성곱으로 텍스트 의미 패턴 포착)
- 본 구현: Steam 리뷰 데이터에 동일 구조 적용 → 리뷰 유용성(votes_up) 회귀

<hr>

<h2>🔧Using tool</h2>

* <code>Python 3.10-3.12</code>
* <code>tensorflow==2.18.1</code>
* <code>gensim>=4.3.0</code>
* <code>nltk>=3.8.0</code>
* <code>scikit-learn>=1.3</code>
* <code>numpy&lt;2.0</code>
* <code>pandas>=2.0</code>
* <code>scipy</code>
* <code>tqdm</code>

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
      NLTK tokenizer 토큰화 → 100차원 Word2Vec 초기화 Embedding layer 통과 → 임베딩 행렬 구성


   <h3>2) Parallel 1D Conv (3 branch)</h3>

   * 합성곱 구조:
      동일 임베딩 행렬에 kernel size 1, 2, 3의 Conv1D 3개 병렬 적용
   * 필터 구성:
      각 branch당 100개 필터, ReLU 활성화 함수 적용
   * 추출 패턴:
      개별 단어 / 인접 두 단어 / 세 단어 범위의 지역적 패턴 동시 추출

   <h3>3) GlobalMaxPooling1D + Concat</h3>

   * 풀링 방식:
      각 branch 출력에 GlobalMaxPooling1D 적용 → 최대 활성화 값 추출
   * 효과:
      각 필터의 최강 반응 위치 신호만 단일 벡터로 압축
   * 결합:
      100차원 pooled 벡터 3개 concat → 300차원 통합 벡터 구성


   <h3>4) Regression Head</h3>

   * 적용 순서:
      Dropout → Dense(32, ReLU) → Dropout → Dense(1, linear)
   * 역할:
      중간 Dense layer: 차원 축소 및 비선형성 부여 / 마지막 linear layer: 유용성 점수 출력

<hr>
