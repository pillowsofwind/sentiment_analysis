# mini_sentiment_analysis
Sentiment analysis using CNN&amp;RNN ,with Pytorch

---

## 使用方法

1. 下载训练集（以准备好）
2. 执行

~~~
python main.py [--model] --train --test -e 50
~~~

参数解释：

| --model | 选择训练、测试的模型（默认参数0）：0 Word Average Model；1 RNN(LSTM) Model；2 CNN Model |
| ------- | ------------------------------------------------------------ |
| --train | 训练                                                         |
| --test  | 测试（可与训练模式同时输入）                                 |
| -i      | 接受控制台的输入（一句话）                                   |
| -e      | 训练的epoch数目（默认参数10）                                |

