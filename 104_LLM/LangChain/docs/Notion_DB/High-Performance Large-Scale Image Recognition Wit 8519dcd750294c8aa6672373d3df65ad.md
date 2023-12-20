# High-Performance Large-Scale Image Recognition Without Normalization

### Authors

Andrew Brock 

Soham De

Samuel L. Smith

Karen Simonyan

DeepMind

### Prerequisites

1. Batch Normalization 
    
    [https://arxiv.org/pdf/1502.03167.pdf](https://arxiv.org/pdf/1502.03167.pdf)
    
2. Brock et al. 2021. Characterizing Signal Propagation to Close the Performance Gap in Unnormalized ResNets ([https://arxiv.org/pdf/2101.08692.pdf](https://arxiv.org/pdf/2101.08692.pdf))
    - 상세 내용
        1. **Contribution**
            - forward pass의 시그널 전파를 시각화하는 분석 도구 제안 (SPP, Signal Propagation Plot)
            - BatchNorm 레이어 없이 높은 성능을 발휘하는 ResNet 설계
            
        2. **주제 논문과의 공통점 & 차이점**
            - 공통점
                - Batch Normalization의 문제를 지적 - 학습 데이터 간의 독립성 훼손, 계산량 및 메모리 과부하, 예상치 못한 버그 유발, batch size에 따라 정확도 달라짐, 학습과 추론 간의 갭
                - Batch Normalization의 장점 나열 - loss surface를 스무스하게 해줌, 높은 learning rate로 학습 가능, 미니배치마다 다른 statistics로 인한 정규화 효과, skip connection 사용시 forward pass의 신호 전파가 잘 이루어지도록 해줌
                - BatchNorm 사용하지 않음
            - 차이점:
                - BatchNorm의 역할을 대신하는 Scaled Weight Standardization 제안
                - 근데 SOTA보다 못함.. 그래서 추후에 (1달 뒤에) Adaptive Gradient Clipping 제안
                
        3. **SPP (Signal Propagation Plot)**
            - SPP는 forward pass의 시그널 전파만을 시각화하지만, 이전 연구들에 의하면 forward pass의 신호 전파가 제대로 이루어지는 한 backward pass의 신호가 explode하거나 vanish하지 않음. 따라서 뉴럴넷을 설계하는 데 유용하게 활용 가능
            
            ![High-Performance%20Large-Scale%20Image%20Recognition%20Wit%208519dcd750294c8aa6672373d3df65ad/2021-03-09_221623.png](High-Performance%20Large-Scale%20Image%20Recognition%20Wit%208519dcd750294c8aa6672373d3df65ad/2021-03-09_221623.png)
            
            - BatchNorm, ReLU activation, He initialization 사용하여 BN-ReLU-Conv (일반적으로 사용되는 순서) 및 ReLU-BN-Conv의 forward pass 전파를 시각화
                
                ![High-Performance%20Large-Scale%20Image%20Recognition%20Wit%208519dcd750294c8aa6672373d3df65ad/2021-03-09_222814.png](High-Performance%20Large-Scale%20Image%20Recognition%20Wit%208519dcd750294c8aa6672373d3df65ad/2021-03-09_222814.png)
                
            - Average Channel Variance의 경우, 각 stage의 depth가 깊어질수록 선형적으로 증가하다가 각 trainsition block에서 1로 초기화됨
            - 용어 정리 - transition block? stage?
            
            ![High-Performance%20Large-Scale%20Image%20Recognition%20Wit%208519dcd750294c8aa6672373d3df65ad/2021-03-11_215635.png](High-Performance%20Large-Scale%20Image%20Recognition%20Wit%208519dcd750294c8aa6672373d3df65ad/2021-03-11_215635.png)
            
        4. **Scaled Weight Standardization**
3. Gradient Clipping
    - 상세내용
        
        논문 링크: [https://arxiv.org/pdf/1211.5063.pdf](https://arxiv.org/pdf/1211.5063.pdf)
        
        - 해결하고자 했던 문제
            
            RNN 계열의 모델 등 딥러닝 모델이 가지는 vanishing, exploding gradient를 극복
            
        - 제안한 해결 방법
            
            일정 threshold를 정하고, gradient의  norm이 그 threshold 보다 크면, 모델의 gradient를 threshold와 곱하고, norm으로 나눔 
            
            ![High-Performance%20Large-Scale%20Image%20Recognition%20Wit%208519dcd750294c8aa6672373d3df65ad/Untitled.png](High-Performance%20Large-Scale%20Image%20Recognition%20Wit%208519dcd750294c8aa6672373d3df65ad/Untitled.png)
            
            - 놓치고 있던 점 (김병현)
                - Gradient의 norm의 계산이 모든 레이어의 gradient로 계산된 값, global 한 값이었다는 점을 제대로 이해하지 못하고 있었음 
                —> Adaptive gradient clipping과 다른 점 
                그래서 AGC보다 GC가 더 큰 threshold로 clipping을 하는 것으로 판단됨
                - 따라서 Batch-Norm을 대체하기 위한 시도로 제안된 AGC는 레이어 단위로 gradient를 clipping 해줄 필요가 있었던 것으로 보임
            - Pseudo 코드의 Pseudo 코드(?)
                
                ```python
                1. 레이어 전체의 gradient에 대하여 norm을 구함 (보통 L2 Norm)
                2. gradient의 norm 이 threshold를 넘는지 넘지 않는지 판단 
                
                3-1. (넘는 경우) 
                	(Pytorch나 Tensorflow 에서는 Iterable하게 
                   각 레이어의 gradient에 접근할 수 있으므로) 
                  각 레이어의 gradient에 for loop으로 접근하여 threshold를 곱하고 norm으로 나눠줌 
                
                2. (안넘는경우) 
                	아무 동작 안함 
                
                ```
                
        - 효고
            
            점선은 gradient가 rescale 됬을 때 나타나는 gradient의 방향을 나타냄 
            
            ![High-Performance%20Large-Scale%20Image%20Recognition%20Wit%208519dcd750294c8aa6672373d3df65ad/Untitled%201.png](High-Performance%20Large-Scale%20Image%20Recognition%20Wit%208519dcd750294c8aa6672373d3df65ad/Untitled%201.png)
            
4. Contrastive Learning
    - 내용
        - Pairwise Loss / Triplet Loss
            
            ![High-Performance%20Large-Scale%20Image%20Recognition%20Wit%208519dcd750294c8aa6672373d3df65ad/Untitled%202.png](High-Performance%20Large-Scale%20Image%20Recognition%20Wit%208519dcd750294c8aa6672373d3df65ad/Untitled%202.png)
            
            - 위 그림은 Triplet Loss 설명. Pairwise Loss는 Positive와 Negative만 존재하는 Loss라고 볼 수 있음
            - 일반적인 Loss는 단일 Image만 가지고 Loss를 계산하여 Parameter를 Update.
            - Pair-Wise Loss는 2개, Triplet은 3개, Quadruplet Loss는 4개
            - Pair Wise Loss는 Anchor, Positive, Negative, 즉 Positive 2개와 Negative 1개를 추출차여 유클리디안 거리를 조절하는 방식
        - Contrastive Learning은 Pairwise Loss와 동일하게 작동하는 방법
        - SimCLR
            - Constrastive Learning을 사용한 Self-Supervised Learning 논문
            - Negative Sample을 사용하지 않고, Batch Size를 크게 잡아서 Input 이미지를 제외한 나머지 Batch들이 Negative이라고 가정 (Batch Size를 키워서 Negative Sample이 있을 가능성을 높임)
            - 큰 Batch가 사용되므로 (8192개) Multi GPU를 사용해야 하고, 그 과정에서 Mini-batch 단위로 평균과 분산을 다 계산해서 합침 (GPU 끼리의 정보이동 및 Aggregation 과정이 필요)
            - 일반적인 BN은 Activation Function전에 적용하는데, 위 논문은 Activation 이후에 적용함
        - Batch와 관련된 참고 논문 : Understanding self-supervised and contrastive learning with "Bootstrap Your Own Latent" (BYOL)([요약](https://generallyintelligent.ai/understanding-self-supervised-contrastive-learning.html))
            - 리뷰하고 있는 논문에서 언급되어 있지 않은걸 보아 위 논문의 내용이 정확하게 일치 하지 않을 수 있지만 유사한 내용이 있음
            - 결론적으로, Contrastive Learning 방법들(MoCo, SimCLR)은 Negative Sample들을 사용하지 않는데, BN이 Contrastive Learning 방식을 암시적으로도입해준다.  (실험을 통해 증명함)
            - 결론중에서, BN을 사용하면 Network Output은 Pure Function을 학습하지 않는다고 한다. 때문에 BN을 피하는 것이 좋다는 언급이 있다.
                - (의견) Pure한 Function을 학습하지 않는다는 뜻은. BN을 통해서 네트워크가 학습한 어떠한 수식의 원형을 알 수 없다는 것을 의미하는 것 같음. Mean shift 등의 계산으로 Input에 어떠한 변형이 항상 가해지고, 그 변형들은 전체 Dataset에 따라 다르기 때문에 Input에 대한 Output의 관계가 항상 일정하지 않음.
            
5. BN과 관련된 참고[자료](https://www.alexirpan.com/2017/04/26/perils-batch-norm.html) : BN 사용에 대한 불합리성

## 1. Introduction

- **배치 정규화는 3가지 단점이 존재한다.**
    - "First, it is a surprisingly expensive computational primitive, which incurs memory overhead (Rota Bul`o et al., 2018), and significantly increases the time required to evaluate the  gradient in some networks (Gitman & Ginsburg, 2017)"
    
        ⇒ 계산 과부하, 학습 시에는 평균 및 분산을 구해야 하고, 이동 평균과 분산도 같이 계산해야 한다.
    
    - "Second, it introduces a discrepancy between the behavior of the model during training and at inference time (Summers & Dinneen, 2019; Singh & Shrivastava, 2019)"
    
        ⇒ 학습과 추론 시간에 차이가 발생한다.
    
    - "Third, batch normalization breaks the independence between training examples in the minibatch"
    
        ⇒ 미니 배치의 독립성을 깨트린다. 
    

- "practitioners have found that batch normalized networks are often difficult to replicate precisely on different hardware, and batch normalization is often the cause of subtle implementation errors, especially during distributed training(Pham et al., 2019)"

      **⇒ (실무자들의 의견에 의하면) 배치 정규화를 하면 HW에 따라 결과가 매번 달랐으며, 특히 분산 학습에서 미세한 구현 에러가 발생하였다.**

- "the interaction between training examples in a batch enables the network to ‘cheat’ certain loss functions. For example, batch normalization requires specific care to prevent information leakage in some contrastive learning algorithms"

    **⇒ 미치 배치와의 상호 작용이(중간에 $\gamma$, $\beta$ 를 학습 하는 것을 뜻하는 듯) 로스 함수를 속인다(?)  예를 들면 contrastive learning 알고리즘에서 정보의 누수를 막기 위해 주의를 기울여야 한다.  특히, 언어 모델에서 (배치 정규화에 의한 정보 누수가) 빈번하며 배치 사이즈, 배치의 분산에  따라 결과가 상이하다.**

- 분산 학습을 하게 되면, 미니 배치의 이동평균을 구하기 힘들어 진다([link](https://www.youtube.com/watch?v=rNkHjZtH0RQ))
    
    ![High-Performance%20Large-Scale%20Image%20Recognition%20Wit%208519dcd750294c8aa6672373d3df65ad/Untitled%203.png](High-Performance%20Large-Scale%20Image%20Recognition%20Wit%208519dcd750294c8aa6672373d3df65ad/Untitled%203.png)
    

- 몇몇 정규화를 없애는 시도가 존재하였음.
    - Suppressing the scale of the hidden activations on the residual branch (Huanget al., 2017; Jacot et al., 2019).
    - (활성화 함수가 유발하는) Mean Shift를 제거하기 위해 Scaled Weight Standardization 도입(Qiao et al., 2019)
    

본 논문에서 제안하는 것

1. Adaptive Gradient Clipping (AGC)
    - (기울기 Norm / 매개변수 Norm) 비율로 Gradient Clpping을 한다.
    - AGC를 통해 배치가 크든, 데이터 확장을 하던, 강인함을 보임을 증명

  2.  NFNet-F1, Similar accuracy to EfficientNet-B7 while being 8.7 faster to train(86.5% top-1)

  3. Fine-tuning 시에 BN 적용한 네트워크 비해 높은 Validation Acc를 달성(89.2% top-1)

- Introduction 상세 정리
    - 최근 Computer Vision분야의 Model들 대부분은 Batch Normalization으로 학습된 Deep Residual Networks이다.
    - 이 두가지 구조의 결합은 높은 성능을 내는 매우 깊은 Network를 학습시킬수 있도록 하였다.
    - Batch Normalization은 또한 Loss Landscape를 Smoothen 하고, 이로인해 큰 Learning Rate와 큰 batch Size에서 안정적으로 학습할 수 있으며 Regularzing 효과 또한 얻을 수 있다.
    - 하지만 Batch Normalization은 3가지 큰 단점이 존재한다.
    - 첫째로 Computation Cost가 매우 높고 Memory Overhead를 야기하며 시간이 많이 든다.
    - 둘째로,  튜닝이 필요한 Hiddin Hyper Parameter를 도입해야 한다는 Training과 Inference간의 Model 차이를 야기한다.
    - 제일중요한 세번째는 Batch Normalization은 Mini Batch간의 독립성을 깨트린다.
        - (의견)Local Minimun Loss 문제 등이 Batch로 학습 시 일어나고는 한다. 이는 모든 데이터셋이 하나로 묶이기 때문인데, 각 이미지를 독립적으로 보지않고 하나의 데이터로 본 뒤 모든 Traning Data를 순회한 뒤 한번의 Update를 하기 때문이다.
        - 하지만 Stochastic 같은 경우에는 하나의 Image만을 가지고 가중치를 업데이트 하게되는데, 이는 이미지 끼리 독립적이라고 볼 수 있다. 이러한 경우에는 하나의 이미지에 최적화된 파라메터를, 다른 이미지에 의해 Loss가 크게 나와서 Local Minimum에서 빠져나와 업데이트가 가능하게 해준다. 따라서 학습 데이터끼리는 독립적이여야 가중치 업데이트에 유리하다고 볼 수 있다.
    - 이 세가지 특징은 부정적인 결과를 초래한다.
    - 예를들어 batch normalized networks는 종종 다른 Hardware에서 복제하기 어렵고, Batch Normalization은 구현하는데 있어 Error를 야기하기도 한다. 
    (개발자들이 사용해보기 어렵다는 뜻)
    - 더불어 Batch Normalization은 Bach간의 상호작용으로 인해 Network가 특정 Loss 함수를 'Cheat' 할수 있기 때문에 어떤 Task에서는 사용이 불가능하다.
    - 예를들어 Batch Normalization은 Constrative Learning Algorithm에서 정보 손실을 방지하기위한 특정한 방법이 필요하다.
    - 이것은 Sequence Model에서도 주로 고려되고 있고, 이 때문에 Language Model에서 대체 Normalizer를 사용하는 사례도 있다.
    - Batch-Normalized Network는 Batch가 학습간에 큰 Variance를 갖는다면 성능이 저하될 수 있다.
    - 결과적으로 Batch Normalization의 성능은 Batch Size에 민감하고 Batch Size가 매우 작으면 성능이 매우 나쁘며 이는 유한한 Hardware로 인해 성능이 제한된다고 할 수 있다.
    - 그래서 Batch Normalization이 최근 몇년간 Deep Learning 커뮤니티에서 인기가 있었음에도 불구하고 장기적인 관점에서 좋지 않은 방법이라고 생각하였다.
    - 따라서 높은 Accuracy와 넓은 적용 범위를 보장할 수 있는 대체 방식을 찾아야 한다.
    - 많은 수의 대체 Normalizer가 제안되었음에도 불구하고 이러한 대체 Normalizer들은 종종 낮은 Accuracy와 Inference 시 Computation Cost가 증가하는 경우와 같은 단점이 존재한다.
    - 다행히도, 최근에 두가지 유망한 연구가 진행되었다.
    - 첫번째 연구는 Training간에 Batch Normalizaion의 이점을 연구하였고, 두번째는 ResNet을 학습할 때 Normalizer 없이 학습하는 방법이다.
    - 이러한 연구들의 주된 테마는 Normalization없이 Residual Branch의 Hidden Activation을 제한하여 매우 깊은 ResNet을 학습하는 것이 가능하다는 것이다.
    - 가장 간단한 방법은 학습 가능한 Scalar를 각 Residual Branch 뒤에 붙이고, 0으로 초기화 하는 방법이다.
    - 하지만 이러한 방법만 사용하는 것은 성능을 내는데 충분하지 않다.
    - 또다른 문제는 ReLU Activation이 Mean Shift를 초래한다는 것이다. Mean Shift는 다른 Training Data의 Hidden Activation들이 Netowork가 깊어질수록 연관되게 한다.
    - 최근의 연구에서 Normalizer-Free ResNet이 제안되었고, Initialization 시 Residual Branch를 억제하고 Mean Shift를 제거하기 위해 Scaled Weight Standardization을 적용하였다.
    - 추가적인 Regularization으로, 이러한 Unnormalized Network는 Batch-Normalized된 Network와 유사한 성능을 내었지만, 큰 Batch에서 여전히 불안하고 현재 SOTA인 EfficientNet보다 성능이 좋지 않다.
    - 본 논문의 Contribution은 아래와 같다.
        - Adatpive Gradient Clipping(AGC) 제안하고, 이는 Gradient Norm / Parameter Norm의 Unit-wise ratio를 바탕으로 Gradients를 Clip한다. 그리고 AGC가 큰 Batch와 강한 Aug에서 Normalizer-Free Network를 어떻게 만드는지 설명한다.
        - Normalizer-Free Renet을 설계하고, NFNet이라고 부름. SOTA달성했고, NFNet-F1은 EfficientNet-B7과 비슷한 성능을 8.7배 빠른 Traning으로 나타내었고, 큰 모델은 SOTA 달성(86.5%)
        
        ![High-Performance%20Large-Scale%20Image%20Recognition%20Wit%208519dcd750294c8aa6672373d3df65ad/Untitled%204.png](High-Performance%20Large-Scale%20Image%20Recognition%20Wit%208519dcd750294c8aa6672373d3df65ad/Untitled%204.png)
        
        - NFNet이 Pretrain 후 Fine Tuning 시 Batch Normalized Network보다 Validation Acc가 더 높다는 것을 설명. (89.2%)

## 2. Understanding Batch Normalization

![High-Performance%20Large-Scale%20Image%20Recognition%20Wit%208519dcd750294c8aa6672373d3df65ad/Untitled%205.png](High-Performance%20Large-Scale%20Image%20Recognition%20Wit%208519dcd750294c8aa6672373d3df65ad/Untitled%205.png)

- 보통 Fully Connected 혹은 Convolutional 계층 앞에 위치하고, 비선형 함수 뒤에 위치한다([link](http://cs231n.stanford.edu/slides/2018/cs231n_2018_lecture06.pdf))
- **학습**시에
    - 미니 배치의 **평균**과 **분산**을 구해서 평균 '0', 분산 '1' 값을 갖게 한다.
    - $\gamma$, $\beta$ 를 학습한다.
    - '이동 평균'과 '이동 분산(?)'을 계산해 놓는다([link](http://cs231n.stanford.edu/slides/2018/cs231n_2018_lecture07.pdf)).
- **테스트** 시에
    - 미니 배치의 **평균**과 **분산**을 구하지 않는다.
    - 위에서 구한 'Moving Average(이동 평균)'과 'Unbiased Variance Estimate(이동 분산)'으로 정규화 한다.
- 배치 정규화의 효과
    - Loss의 경사를 부드럽게 한다(smoothens the loss landscape)
    
         = Improve gradient flow through the network(cs231n)
    
    - (활성화 함수에 의해 유발 된) Mean-Shift를 줄여준다
    - 배치 사이즈가 커도, 학습이 안정적이게 된다.
    

- 상세 정리
    
    ### Batch Normalization downscales the Residual Banch
    
    - Skip Connection과 Batch Normalization의 조합은 Deep 하게 학습할 수 있게 하였다.
    - 이러한 이점은 Batch Normalization이 Residual Brance에 위치해 있을 때 발생하며, Initialization 시 Residual Brench에 존재하는 Hidden Activation의 크기를 감소시킨다.
        - Hidden Activation은 그냥 Hidden Layer를 의미한다고 생각할 수 있을 지 ?
    - 이러한 이론은 training 시 빠르고 효과적으로 최적화 할 수 있게 한다.
    
    ### Batch Normalization Eliminates mean-shift
    
    - ReLU나 GELU같은 Activation  Function은 0이 아닌 평균이 0이 아니다.
    - 결과적으로 Input Feature간의 내적이 0에 가까울 지라도 높은 성능의 독립된 Non-Linearity 직후의 Training Data들의 Normalizer-Free Resnet Activation간의 내적은 크고 양의 값이다.
    (평균이 0이 아니므로 내적하면 0이 아닌 값이 아닌 양수의 값이 나온다는 뜻)
    - 이러한 문제는 네트워크 깊이가 증가함에 따라 폭합적으로 작용하며 네트워크의 깊이에 비례하여 다른 Training Data간의 Activation에 'Meah-Shift'를 초래한다. 이는 Deep Network가 초기화시 모든 데이터를 단일 Label로 예측하는 문제를 야기한다.
    - Batch Normalization은 Activation의 평균을 0로 만들고 mean shift를 제거한다.
    
    ### Batch Normalization has a regularizing Effect
    
    - Batch Normalization이 batch statistics의 noise 때문에 Regularization 효과가 있는 것은 널리 믿고 있는 사실이다.
        - (Statistics : Mean / Variance)
    - 이러한 관점에 일치하여, Batch Normalized Network의 test Acc는 batch Size를 튜닝하여 올릴 수 있다.
    
    ### Batch Normalization allows efficient large-batch training
    
    - Batch Normaliztion은 loss landscape를 Smoothen 하고, 이러한 것은 큰 Learning Rate에서도 안정적이게 한다.
    - 이러한 특징이 batch size가 작을때는 적용되지 않더라도, 큰 Batch Size에서 학습할 때에는 충분히 필요하다.
    - 고정된 epoch에서 큰 batch 학습이 test ACC가 높지 않더라도, 몇몇의 파라메터 업데이트로 test ACC를 올릴수 있다.

## 3. Towards Removing Batch Normalization

- **"Normalizer-Free ResNets(NF-ResNets)" (Brock et al., 2021)**
    - 다음과 같은 형태의 Residual Block을 취하였다.
        
        ![High-Performance%20Large-Scale%20Image%20Recognition%20Wit%208519dcd750294c8aa6672373d3df65ad/Untitled%206.png](High-Performance%20Large-Scale%20Image%20Recognition%20Wit%208519dcd750294c8aa6672373d3df65ad/Untitled%206.png)
        
    - 정규화 해주는 것과 비슷.
- **Scaled Weight Standardization (Qiao et al., 2019)**
    - Mean-Shift를 없에기 위하여 가중치를 정규화 해준다.
    
    ![High-Performance%20Large-Scale%20Image%20Recognition%20Wit%208519dcd750294c8aa6672373d3df65ad/Untitled%207.png](High-Performance%20Large-Scale%20Image%20Recognition%20Wit%208519dcd750294c8aa6672373d3df65ad/Untitled%207.png)
    
    - 활성화 함수도 $\gamma$ 로 Scaled 된다.
- 그외
    - Dropout (Srivastava et al.,2014
    - Stochastic Depth (Huang et al., 2016))
- 이러한 시도들은 배치 사이즈가 작을 때는 배치 정규화 적용한 것에 비해 성능이 좋았지만, 배치가 클때는(e.g 4096 이상) 배치 정규화 성능을 따라 잡기 힘들었다.

- 상세정리
    - 많은 저자들은 위에서 언급한 Batch Normalization의 이점을 하나 혹은 여러개를 가져가며  Normalization 없이 Deep Resnet을 학습시키려고 노력하고있다.
    - 이러한 연구의 대부분은 학습가능한 scalar 또는 작은 상수를 도입해  Initialization에서 Residual Branch의 activation의 크기를 제한(억눌러)한다.
    - 더불어 몇몇의 연구는 unnormalized Restnet이 추가적인 regularization을 통해 성능이 향상될 수 있음을 확인하였다.
    - 하지만 단지 이 두개의 batch normalization의 이점만을 가지고 간다는 것은 높은 test Acc를 도달하는 것에 충분하지 않다.
    - 본 연구에서는 Normalizer-Free Resnet(NF-ResNEts)의 구조를 차용한다. NF-Resnet은 $h_{i+1}=h_{i}+\alpha f_{i}(h_{i}/\beta_{i})$형태의 Residual Bock을 사용한다.
    - $h_{i}$는 i번째 Residual Block이고, $f_{i}$는 i번째 Residual Brench에서 계산되는 Function이다.
    - $f_{i}$은 Initialization될 때 모든 i에 대해서 $Var(f_{i}(z))=Var(z)$처럼 Variance를 유지하도록 Parameterized된다.
        - 입력값과 출력의 variance가 동일 하도록 한다 ?
    - Scalar값 $\alpha$는 각 Residual Block 이후 activation의 variance가 증가하는 비율을 정해주고, 보통 0.2같이 작은 값을 사용한다.
        - 0.2를 사용하면 증가하는 속도를 감소시키는 형태 (가속도를 감소, 속도를 감소시키는 것은 아님)
    - Scalar값 $\beta_{i}$는 i번째 Residual Block의 표준편차를 예측함으로서 결정된다.
        - $Var(h_{i+1})=Var(h_{i})+\alpha^{2}$일때 $\beta_{i}=\sqrt{Var(h_{i})}$로 정의됨.
    - Skip Path가 $h_{i}/{\beta_{i}}$로 DownScale이 일어나는 Transition Block에서는 Variance가 Transition Block 이후 $h_{i+1}=1+\alpha^{2}$로 Reset된다.
        - (Transition Block : Pooling)
    - Squeeze-excite Layer의 Output은 Factor 2를 곱한다.
        - SNEet 구조 참조
            
            ![High-Performance%20Large-Scale%20Image%20Recognition%20Wit%208519dcd750294c8aa6672373d3df65ad/Untitled%208.png](High-Performance%20Large-Scale%20Image%20Recognition%20Wit%208519dcd750294c8aa6672373d3df65ad/Untitled%208.png)
            
    - 경험적으로, 학습가능한 scalar를 residual Brench 뒤에 붙이는 것이 이점이 있는 것 또한 증명되었다.
    - 또한 Scaled Weight Standardization을 통해 Hidden Activation의 Mean-SHift를 방지한다.
    - 이러한 테크닉은 Conv Layer를 $\hat{W}_{ij}={{W_{ij}-\mu_{i}}\over{\sqrt{N}\sigma_{i}}}$로 Reparameterized 한다.
    - 이때,  각 값들은 아래와 같음.
        
        ![High-Performance%20Large-Scale%20Image%20Recognition%20Wit%208519dcd750294c8aa6672373d3df65ad/Untitled%209.png](High-Performance%20Large-Scale%20Image%20Recognition%20Wit%208519dcd750294c8aa6672373d3df65ad/Untitled%209.png)
        
    - N은 Fan-in(입력의 갯수)
    - Activation Function은 또한 Non-linearity scalar gain값 $\gamma$로 조정된다. $\gamma$로 조정된 activation function과 Scaled Wegiht Standardized layer의 조합은 variance를 유지시킨다.
    - ReLU에서, $\gamma=\sqrt{2/(1-(1/\pi))}$이다.
    - 위 설명이 기술된 논문에서 다른 Non-linearity에서 $\gamma$를 계산하는 방법이 기술되어 있다.
    - 추가적인 Regulrization과 Stochastic Depth로, 배치사이즈 1024로 학습한 Normalizer-Free Renet은 Test Acc가 Batch Normalized Resnet과 유사한 성능을 보였다.
    - 하지만 4096 또는 더 높은 수치에서 성능이 안좋았고, 결정적으로 EfficientNet보다 성능이 안좋았다.

## 4. Adaptive Gradient Clipping for Efficient Large-Batch Training

**Gradient clipping** is typically performed by constraining the norm of the gradient (Pascanu et al., 2013).

$$
G \rightarrow\left\{\begin{array}{ll}\lambda \frac{G}{\|G\|} & \text { if }\|G\|>\lambda \\ G & \text { otherwise. }\end{array}\right.
$$

The clipping threshold λ is a hyper-parameter which must be tuned.

To overcome this issue, we introduce “Adaptive Gradient Clipping” (AGC), which we now describe.

$$
G_{i}^{\ell} \rightarrow\left\{\begin{array}{ll}\lambda \frac{\left\|W_{i}^{\ell}\right\|_{F}^{\star}}{\left\|G_{i}^{\ell}\right\|_{F}} G_{i}^{\ell} & \text { if } \frac{\left\|G_{i}^{\ell}\right\|_{F}}{\left\|W_{i}^{\ell}\right\|_{F}^{\star}}>\lambda \\G_{i}^{\ell} & \text { otherwise. }\end{array}\right.
$$

where 

$$
\mid W_{i} \|_{F}^{\star}=\max \left(\left\|W_{i}\right\|_{F}, \epsilon\right)
$$

Let   $W^{\ell} \in \mathbb{R}^{N \times M}$ denote the weight matrix of the th layer,    $G^{\ell} \in \mathbb{R}^{N \times M}$ denote the gradient with respect to W, M and · F denote the Frobenius norm.

### Motivation

배치 정규화를 사용하지 않고도, 큰 사이즈의 배치를 사용하기 위해 Gradient Clipping 전략을 참고 하였다.

- "조건이 좋지 못한 Loss 함수의 경우(*경사가 거친 모양을 뜻하는 듯*) 혹은 배치사이즈가 클때는 최적의 lr은 maximum stable learning rate에 의해 제약 사항이 생긴다" (?)

- Gradient Clipping 이란
    
    ![High-Performance%20Large-Scale%20Image%20Recognition%20Wit%208519dcd750294c8aa6672373d3df65ad/Untitled%2010.png](High-Performance%20Large-Scale%20Image%20Recognition%20Wit%208519dcd750294c8aa6672373d3df65ad/Untitled%2010.png)
    
- Gradient Clipping의 한계
    - Gradient Clipping이 배치 사이즈를 늘리면서 학습이 가능하게는 하지만, 학습의 안정성은 다소 떨어진다(Threshold, 모델 깊이, lr 등에 민감함)

### Adaptive Gradient Clipping

- 용어 정리
    
    $$
    \begin{aligned}&W^{l} \in R^{N \times M}: l^{t h} \text { 번째 계층의 가중치 행렬 }\\&G^{l} \in R^{N \times M}: W^{l} \text { 에 대응하는 기울기 }\\&\text { || } \cdot \|_{F} \text { 는 Frobenius norm }\left(L^{2}\right)\\&\left\|W^{l}\right\|_{F}=\sqrt{\sum_{i}^{N} \sum_{j}^{M}\left(W_{i, j}^{l}\right)^{2}}\end{aligned}
    $$
    
- 비율 $\frac{\left \| G^l \right \|}{\left \| W^l \right \|}$이 경사 하강법 Step에서 원래 가중치 W를 얼마나 변경하는지를 나타 내는 단위가 될 수 있다는 점에 영감을 얻었다.
- 예를 들어 경사 하강 법을 이용해 (모멘텀 없이) 학습 시킬 경우 $\frac{\left \| \Delta W^l \right \|}{\left \| W^l \right \|}$= h$\frac{\left \| G^l \right \|}{\left \| W^l \right \|}$ 다음과 같이 나타낼 수 있고, $l^{th}$ 번째 계층의 가중치 업데이트는 다음과 같이 주어진다.

      $\Delta{W^l} = -hG^l$, h는 learning rate.

- $l^{th}$ 번째 계층의 기울기는 다음과 같이 Clipping 될 수 있다.
    
    $$
    G_{i}^{\ell} \rightarrow\left\{\begin{array}{ll}\lambda \frac{\left\|W_{i}^{\ell}\right\|_{F}^{\star}}{\left\|G_{i}^{\ell}\right\|_{F}} G_{i}^{\ell} & \text { if } \frac{\left\|G_{i}^{\ell}\right\|_{F}}{\left\|W_{i}^{\ell}\right\|_{F}^{\star}}>\lambda \\G_{i}^{\ell} & \text { otherwise. }\end{array}\right.
    $$
    
- 하이퍼 파라미터 $\lambda$는 다음과 같이 정의한다.
    - 
        
        ![High-Performance%20Large-Scale%20Image%20Recognition%20Wit%208519dcd750294c8aa6672373d3df65ad/Untitled%2011.png](High-Performance%20Large-Scale%20Image%20Recognition%20Wit%208519dcd750294c8aa6672373d3df65ad/Untitled%2011.png)
        
    
    "최적의 파라미터 값은 옵티마이저, 배치 사이즈에 따라 조금씩 다르다. 하지만, 큰 배치의 경우는 값이 작아야 한다"
    
- 
- AGC를 이용하면, 큰 배치사이즈(e.g. 4096) 과 데이터 확장(Augmentation)에도 강인함을 보였다.
    
    기존의 NF-ResNet은 배치 사이즈가 커지면, 정확도가 확 떨어지지만, AGC를 더하면 정확도가 유지 된다.
    
    ![High-Performance%20Large-Scale%20Image%20Recognition%20Wit%208519dcd750294c8aa6672373d3df65ad/Untitled%2012.png](High-Performance%20Large-Scale%20Image%20Recognition%20Wit%208519dcd750294c8aa6672373d3df65ad/Untitled%2012.png)
    
- "AGC can be interpreted as a relaxation of normalized optimizers, which imposes a maximum update size biased on the parameter norm but does not simultaneously impose a lower-bound on the update size or ignore the gradient magnitude"

    ⇒ 기울기의 크기를 고려하면서, 가중치 업데이트 크기를 조정한다. (*라고 해석하면 될까...)*

**(사견) 하이퍼 파라미터 설정을 최대한 피하고, 가중치 값에 따라 Clipping 정도가 정해지는 것을 강조하기 위해 Adaptive라고 이름 붙인 듯 하다**.

### 4.1. Ablations for Adaptive Gradient Clipping (AGC)

(작성중)

## 5. Normalizer-Free Architectures with Improved Accuracy and Training Speed

![High-Performance%20Large-Scale%20Image%20Recognition%20Wit%208519dcd750294c8aa6672373d3df65ad/Untitled%2013.png](High-Performance%20Large-Scale%20Image%20Recognition%20Wit%208519dcd750294c8aa6672373d3df65ad/Untitled%2013.png)

- 최신 SOTA image classification 모델들은 EfficientNet 계열의 논문들임
- EfficientNet은 ResNet-50 보다 10배 적은 FLOPS를 가지고도 비슷한 학습시간을 보임
- 이러한 논문들에서 모델의 구조가 추후 개발되는 가속하드웨어에서 좋은 속도를 보일지 모르나, 현재는 현재 사용하고 있는 하드웨어에서 좋은 성능을 보이는 구조를 찾을 필요가 있음
- 본 논문에서는 최근 SOTA를 달성했던 image classification 모델의 design trend들을 따라 NF한 모델 구조를 찾아보았음
- Baseline model, SE-ResNeXt-D model with GELU activations
    
    [https://arxiv.org/pdf/1709.01507.pdf](https://arxiv.org/pdf/1709.01507.pdf)
    

**First Change** 

- Group width is set to 128 in 3x3 convs.

**Second Change** 

- 균등하게(?) 확장되고 줄어드는 stage 별 depth 모음

![High-Performance%20Large-Scale%20Image%20Recognition%20Wit%208519dcd750294c8aa6672373d3df65ad/Untitled%2014.png](High-Performance%20Large-Scale%20Image%20Recognition%20Wit%208519dcd750294c8aa6672373d3df65ad/Untitled%2014.png)

 **Third Change** 

- 균등하게 확장되는 conv layer channel [256, 512, 1024, 2048] /
- 불규칙한 경우에는 이 옵션만 더 나은 성능을 보임 [256, 512, 1536, 1536].

**NEFNET block 구조** 

![High-Performance%20Large-Scale%20Image%20Recognition%20Wit%208519dcd750294c8aa6672373d3df65ad/Untitled%2015.png](High-Performance%20Large-Scale%20Image%20Recognition%20Wit%208519dcd750294c8aa6672373d3df65ad/Untitled%2015.png)

모델 구성에 대해서 제대로 파악하기 위해서는 Appendix C를 보고 좀 더 파볼 필요가 있을 듯 

- ResNet 구조
    
    ![High-Performance%20Large-Scale%20Image%20Recognition%20Wit%208519dcd750294c8aa6672373d3df65ad/Untitled%2016.png](High-Performance%20Large-Scale%20Image%20Recognition%20Wit%208519dcd750294c8aa6672373d3df65ad/Untitled%2016.png)
    

(김병현) 질문! 원래 Conv layer의 depth를 width라고 불렀었나요...?

![High-Performance%20Large-Scale%20Image%20Recognition%20Wit%208519dcd750294c8aa6672373d3df65ad/Untitled%2017.png](High-Performance%20Large-Scale%20Image%20Recognition%20Wit%208519dcd750294c8aa6672373d3df65ad/Untitled%2017.png)

## 6. Experiments

![High-Performance%20Large-Scale%20Image%20Recognition%20Wit%208519dcd750294c8aa6672373d3df65ad/Untitled%2018.png](High-Performance%20Large-Scale%20Image%20Recognition%20Wit%208519dcd750294c8aa6672373d3df65ad/Untitled%2018.png)

## Conclusion

- 배치 정규화를 적용하지 않고도, 큰 배치에서 배치 정규화를 적용한 모델의 성능을 뛰어 넘는 최초의 모델이다 (*라고 강조함..*)
- 배치 정규화 적용한 모델과 성능은 비슷하면서도빠르게 학습할 수 있다.
- AGC 기법을 적용한 family models을 만들었다.
- 정규화 없는 모델이 (이미지 넷과 같은 모델을 학습 한 후) Finetuning 할때 되려 더 좋은 성능을 나타낸 다는 것을 보였다.

## 총평

- 다소 실무적인 논문이라는 생각이 들었다(실무에서 경험해 봄 직한 배치 정규화 사용후 겪은 단점을 비교적 자세하게 기술)
- 간단한 아이디어지만, Ablation Study 및 타 모델과 성능 비교 자세히 기술.

**Appendix**