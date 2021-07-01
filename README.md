Kaggle Competition
==================
## 1. Titanic ([link](https://www.kaggle.com/c/titanic))
 타이타닉 탑승객의 특성으로 탑승객의 생존 또는 사망을 예측하는 모델을 만드는 이진분류 문제입니다.   
 머신러닝 모델에 그리드서치를 적용해서 하이퍼파라미터 튜닝을 시도하였고 약 80점 이상의 정확도를 보였다.   
 정확도가 높진 않지만 캐글 기준 상위 4퍼센트 정도의 좋은 결과이다.    
 하지만 상대적으로 데이터 사이즈가 작아서 좋은 성능의 모델을 만들기 어려웠다.    
 현재는 가장 좋은 예측을 보여준 XGBOOST를 기준으로 버전업을 하면서 성능을 끌어올리는 중이다.
 
## 2. House Prices ([link](https://www.kaggle.com/c/house-prices-advanced-regression-techniques))
 집값을 예측하는 회귀 모델을 만드는 문제입니다. 처음으로 시도해본 회귀 문제이다.    
 특성이 다양하고 각각의 특성을 이해하고 결측치를 채우는 시간이 필요하다.
 
## 3. Digit Recognizer ([link](https://www.kaggle.com/c/digit-recognizer))
 디지털 이미지로 표현된 숫자를 맞추는 문제입니다. 0~9까지의 숫자를 맞추는 다중분류 문제라고 볼수 있다.    
 이 문제는 딥러닝 기법 중에 하나인 Convolutional Neural Network(CNN)를 사용했다.    
 Kaggle에서 다른 사람들의 Kernel을 참고해서 실제로 Colab에서 적용해봤을때 가장 좋은 결과가 나온 경우가    
 Convolution layer와 maxpooling layer를 조합한 모델에 imagedatagenerator를 이용한 데이터 생성 등을 활용해서    
 LB score 99.4점 정도가 나왔다.
