## Attention Is All You Need (Paper Review & Implementation)

<img width="565" alt="스크린샷 2025-03-05 오후 12 00 47" src="https://github.com/user-attachments/assets/180f161d-d757-43cd-b6be-b7fdb4f69f11" />


## ❗️ 
model.py : Transformer 모델 <br>
data1.py : 데이터 전처리 <br>
train.py : Transformer 모델 학습 <br>
evaluate1.py : Test set에 대한 평가 <br>
inference.py : 새로운 문장들에 대해서 모델 테스트 <br>



## 실행 순서

model.py -> make_model함수에서 하이퍼파라미터를 조절할 수 있습니다.
train.py -> python train.py로 transformer_model을 학습시킬 수 있습니다.
evaluate1.py -> python evaluate1.py로 test셋에 대한 BLEU Score를 확인할 수 있습니다
inference.py -> sentences 속의 번역하고자 하는 문장들을 넣고, python inference.py로 실제 번역 결과를 확인할 수 있습니다.
