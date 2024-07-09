서론
- 오늘날 GPT-2 모델은 Cloud에서 약 $10, 1시간 이내로 학습이 가능하다.
- OpenAI가 2019년에 출간한 [GPT-2](https://d4mucfpksywv.cloudfront.net/better-language-models/language_models_are_unsupervised_multitask_learners.pdf)는 모델 가중치가 공개 되어 있지만 학습 과정의 디테일이 논문에 부족하게 기재되어 있다
- 따라서, 본 강의에서는 [GPT-3](https://arxiv.org/pdf/2005.14165) 논문도 같이 참고해서 GPT-2 모델을 설계해 본다.


본론
- transformer.wte.weight torch.Size([50257, 768])
  - GPT2는 50257 토큰을 가지고 있다. 각 토큰은 768 차원을 가진다.
- transformer.wpe.weight torch.Size([1024, 768])
  - 포지션을 위한 lookup 테이블?? 포지션 임베딩

- Transformer 논문에서 Sinosoial and Consine 진동을 가지는 Positional Embedding은 고정 값이다.
  - 하지만, GPT2에서는 이는 파라미터이고, 학습된다

- GPT-2는 Decoder만 가지고 있다. 또한, Encoder에서 사용하는 Cross-Attention 부분도 없다.
  - 또한, Transformer 논문과 다르게 LayerNorm이 먼저 수행된다.
  - 카파씨가 말하길 residual 모듈은 단독으로 존재하는 것이 낫다고 한다. 즉, 지금 구현은 비 선호.
  - Attention 모듈은 토큰이 서로 소통하게 한다 ==> reduce라 표현
  - mlp 모듈은 토큰 독립적으로 수행한다 ==> map 으로 표현

``` python
class Block(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.ln_1 = nn.LayerNorm(config.n_embd)
        self.attn = CausalSelfAttention(config)
        self.ln_2 = nn.LayerNorm(config.n_embd)
        self.mlp = MLP(config)
    
    def forward(self, x):
        x = x + self.attn(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x
```

- 입력을 (B, T) 사이즈로 만들어야 한다. T는 Maximum Token Length 보다 클 수 없다.


