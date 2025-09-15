
# 【機器學習2021】01
https://youtu.be/Ye018rCVvOo?si=cKmhbW6FT60eZH-F
本來就都會了, 懶得做筆記.
# 【機器學習2021】02 # 預測本頻道觀看人數 (下) - 深度學習基本概念簡介
https://youtu.be/bHcJCp2Fyxs?si=uSrPpHlkR3J96ak3

本堂課一圖以蓋之:
![[Pasted image 20250915154720.png]]

Q: 老師在Slide的P23~26使用了HandWaving的方式證明了任何一維的連續函數都能用如下的HardSigmoid function來近似:
$$
\begin{aligned}
& y=b+\sum_i c_i \operatorname{HardSigmoid}\left(\underline{b_i+w_i x_1}\right) \\
\end{aligned}
$$
但是後面(P30)直接延伸到n維沒給證明，試著自己證明n維的連續函數如何能被下面的函示疊加近似? Hint: 先想在2維，Locally任意的function是否可以被近似? 削蘋果法~
$$
\begin{aligned}
& y=b+\sum_i c_i \operatorname{HardSigmoid}\left(\underline{b_i+\sum_j w_{i j} x_j}\right)
\end{aligned}
$$

Ref: P30把Model從1維延伸到n維如下:
$$
\begin{aligned}
&\text { New Model: More Features }\\
&\begin{aligned}
& y=\underline{b+w x_1} \\
& y=b+\sum_i c_i \operatorname{sigmoid}\left(\underline{b_i+w_i x_1}\right) \\
& y=\underline{b+\sum_j w_j x_j} \\
& y=b+\sum_i c_i \operatorname{sigmoid}\left(\underline{b_i+\sum_j w_{i j} x_j}\right)
\end{aligned}
\end{aligned}
$$
P44 Batch, Epoch: Update所有Batch一次. Update: Batch Size也是一個HyperParameter.
P47~48 
Rectified Linear Unit (ReLU): 
![[Pasted image 20250914194806.png]]
Activation function: sigmoid, HardSigmoid, 使用者自由決定.
Neuron, Neuron Network, hidden layer, Deep Learning: Deep = Many hidden layers.
P57 Why we want “Deep” network, not “Fat” network? (未來課程回答)

課後閱讀:
Basic Introduction: 舊版的Deep Learning介紹
https://www.youtube.com/watch?v=Dr-WRlEFefw
Backpropagation: Computing gradients in an efficient way
https://www.youtube.com/watch?v=ibJpTrp5mcE

Q: 每一層都是把該層的a拿去跟y算Loss(a, y)來逐層的決定(b, w)的最佳化值嗎? 還是每一次updater就同時更新了所有層的(b, w)
![[Pasted image 20250914195912.png]]

# 【機器學習2021】03 # 機器學習任務攻略
https://youtu.be/WeHM2xpYQpw?si=GdNzJ2wa5rqBb355

本課程: 把Learning做好的一些基本招式
Loss太高 => 辨別問題癥結:
Model Bias: Model不夠複雜 => 把Model變得更大。
Optimization: Model已經足夠複雜，是找最佳解的時候出問題，例如可能是Gradient Descent這個方法卡在Local Minimum的問題。
Tips: 任何問題都先從LR或是SVM開始(比較沒有最佳化問題)，再換成複雜的model，跟前面的初階通用Model比較，結果如果比較差，就可以推測複雜的model可能是有最佳化問題。


Data augmentation: 增加(人工產生的)資料。需要有Domain Knowledge當guide，來產生合理的
fake資料。
constrained model: (根據Domain Knowledge)限制你的model自由度: 
•特定形式的function •Less parameters, sharing parameters
• Less features • Early stopping • Regularization • Dropout

Validation:  N-fold Cross Validation
 Mismatch: Training data 跟 Testing date的分布不一樣。HW11
# 【機器學習2021】04~05 # # 類神經網路訓練不起來怎麼辦： 局部最小值 (local minima) 與鞍點 (saddle point) / 批次 (batch) 與動量 (momentum)
https://youtu.be/QW6uINn7uGk?si=DLGu8ZkFLmWX2N_6
https://youtu.be/zzbr1h9sF54?si=BrYqoKnnigO_mrxB
Critical point: Gradient = 0 => Saddle point or local minimum maximum.
檢查Hessian的eigen value來判斷是何種。
算出Hessian => 沿著負Eigenvalue的Eigenvector => Loss變小。但實務上Hessian因為計算量大，通常沒在用。
研究上發現因為Model的參數/維度很大，通常都是卡在SaddlePoint而不是LocalMinimum。 

Smaller batch size and momentum help escape critical points:
Batch:
small Batch => Optimization(training data)，Generalization(test data)比較好!
 => (Training) Time for one epoch 比較快!
課後閱讀: slide上cite一些研究，運用一些技巧來使用large Batch並且避免large Batch的劣勢。

Momentum:
## (Vanilla) Gradient Descent:
![[Pasted image 20250913162615.png]]
Starting at $\boldsymbol{\theta}^{\mathbf{0}}$
Compute gradient $\boldsymbol{g}^{\mathbf{0}}$
Move to $\boldsymbol{\theta}^{\mathbf{1}}=\boldsymbol{\theta}^{\mathbf{0}}-\eta \boldsymbol{g}^{\mathbf{0}}$
Compute gradient $\boldsymbol{g}^{\mathbf{1}}$
Move to $\boldsymbol{\theta}^{\mathbf{2}}=\boldsymbol{\theta}^{\mathbf{1}}-\eta \boldsymbol{g}^{\mathbf{1}}$

## Gradient Descent + Momentum:
![[Pasted image 20250913162740.png]]
Starting at $\boldsymbol{\theta}^{\mathbf{0}}$
Movement $\boldsymbol{m}^{\mathbf{0}}=\mathbf{0}$
Compute gradient $\boldsymbol{g}^{\mathbf{0}}$
Movement $\boldsymbol{m}^{\mathbf{1}}=\lambda \boldsymbol{m}^{\mathbf{0}}-\eta \boldsymbol{g}^{\mathbf{0}}$
Move to $\boldsymbol{\theta}^{\mathbf{1}}=\boldsymbol{\theta}^{\mathbf{0}}+\boldsymbol{m}^{\mathbf{1}}$
Compute gradient $\boldsymbol{g}^{\mathbf{1}}$
Movement $\boldsymbol{m}^{\mathbf{2}}=\lambda \boldsymbol{m}^{\mathbf{1}}-\eta \boldsymbol{g}^{\mathbf{1}}$
Move to $\boldsymbol{\theta}^{\mathbf{2}}=\boldsymbol{\theta}^{\mathbf{1}}+\boldsymbol{m}^{\mathbf{2}}$
Movement not just based on gradient, but previous movement.

# 【機器學習2021】06~ # # 類神經網路訓練不起來怎麼辦：自動調整學習速率 (Learning Rate)

https://youtu.be/HYUXEeh3kwY?si=STswcAHcyFFZIRhL


Gradient Descent往往很難走到norm of Gradient很接近0的位置(critical point)。
GD -> RMSProp -> Adam = RMSProp + Momentum -> + Learning Rate Scheduling
Learning Rate Scheduling: $\boldsymbol{\theta}_i^{\boldsymbol{t}+\boldsymbol{1}} \leftarrow \boldsymbol{\theta}_i^{\boldsymbol{t}}-\frac{\boldsymbol{\eta^t}}{\sigma_i^t} \boldsymbol{g}_i^{\boldsymbol{t}}$
A. Learning Rate Decay: 最後已經接近終點時可以走慢點，避免最後震盪。
![[Pasted image 20250913221016.png]]
B. Warm Up: 有效的原因待研究。可能的解釋: 一開始用來計算$\sigma$的點不夠多，統計誤差大。等經過的點多了，計算出來的$\sigma$統計上更可靠時再走快一點。
![[Pasted image 20250913221025.png]]
課後閱讀: 更多關於的Optimization東西
[TA 補充課] Optimization for Deep Learning
https://www.youtube.com/watch?v=4pUmZ8hXlHM
https://www.youtube.com/watch?v=e03YKGHXnL8
## My Reflections
**GD**:  $\theta_{t+1}=\theta_t-\eta \nabla_\theta J(\theta)=\theta_t-\eta g_t$
如果梯度大 → 走得快；梯度小 → 走得慢。缺點：梯度小時會走得慢，無法收斂到Local minimum(不確定?)
![[Pasted image 20250913214201.png]]
**NGD**(Normalized Gradient Descent): $\theta_{t+1}=\theta_t-\eta \frac{g_t}{\left\|g_t\right\|}$
不管梯度多大，永遠走「固定長度」的一步。。優點：避免梯度爆炸。缺點：忽略梯度大小，收斂慢，可能會在谷底兩側震盪(?)
**RMSProp**: $\boldsymbol{\theta}_i^{\boldsymbol{t}+\boldsymbol{1}} \leftarrow \boldsymbol{\theta}_i^{\boldsymbol{t}}-\frac{\eta}{\sigma_i^t} \boldsymbol{g}_i^{\boldsymbol{t}} \quad \sigma_i^t=\sqrt{\alpha\left(\sigma_i^{t-1}\right)^2+(1-\alpha)\left(\boldsymbol{g}_i^{\boldsymbol{t}}\right)^2}$

Q: 無法理解這樣為什麼比較好。為什麼不是用更直觀的全域修正的方式如下:
$\theta_i^{t+1} \leftarrow \theta_i^t-\frac{\eta}{\sigma^t}, \sigma^t=\sqrt{\alpha\left\|\sigma^{t-1}\right\|^2+(1-\sigma)\left\|g^t\right\|^2}$
簡言之，全域修正($\left\|\boldsymbol{g}^{t-1}\right\|^2$) vs. per-parameter($(g_i^{\boldsymbol{t}})^2$)) 修正: 讓個別的parameter自己去調整為何比較好?

Momentum: $v_t=\beta v_{t-1}+(1-\beta) g_t, \quad \theta_{t+1}=\theta_t-\eta v_t$
這個版本比老師教的多了一個$\beta$自由度，覺得應該是比較好的版本。

# 【機器學習2021】07 # 類神經網路訓練不起來怎麼辦 (四)：損失函數 (Loss) 也可能有影響
https://youtu.be/O2VkP8dJ5FE?si=3p9YEgszC-8Pf1U8

建議閱讀過去更完整版的Classification教學:
https://youtu.be/fZAZUYEeIMg?si=rzaC3rghEbkU201N
https://youtu.be/hSXFuypLukA?si=62MqJx02mXiy-CkV

**x**經NNs(Neural Network)得到**y**(實數)，再追加經過一層softmax得到y'(機率向量)
之後這裡定義Loss as Cross-entropy: $e=-\sum_i \widehat{\boldsymbol{y}}_i \ln \boldsymbol{y}_i^{\prime}$
會比MSE(Mean Square Error)更好。時間有限，課程中只給了Handwaving的解釋。
# 【機器學習2021】08 類神經網路訓練不起來怎麼辦 (五)： 批次標準化 (Batch Normalization) 簡介
https://youtu.be/BABPWOkSbLE?si=Nv7ldK5z5lrZZf5I

Batch normalization: (每個batch進行計算時)對每一層的input xi跟Activation function前(或後)的參數各做一次standard normalization. 若Activation function是sigmoid, 建議要在Activation function前的參數做Normalization比較好.
Inference: 就是testing.
Batch normalization – Testing: testing data要一筆一筆即時算, 沒有batch, 無法計算(σ, µ). Computing the moving average of 𝝁 and 𝝈 of the batches during training 用來給testing data使用.
延伸閱讀: Batch normalization如何用在CNN上? 參考Original paper: https://arxiv.org/abs/1502.03167

Q: 聽無，這段的(γ, β)是啥? 怎麼決定? 為什麼不是用原本的(σ, µ) (layer by layer 的)回推回第一層?
https://youtu.be/BABPWOkSbLE?list=PLJV_el3uVTsMhtt7_Y6sgTHGHp1Vb2P2J&t=1083
$$
\begin{aligned}
& \tilde{z}^i=\frac{z^i-\mu}{\sigma} \\
& \hat{z}^i=\gamma \odot \tilde{z}^i+\beta
\end{aligned}
$$

To learn more: (about Renormalization)
• Batch Renormalization • https://arxiv.org/abs/1702.03275 
• Layer Normalization • https://arxiv.org/abs/1607.06450 
• Instance Normalization • https://arxiv.org/abs/1607.08022 
• Group Normalization • https://arxiv.org/abs/1803.08494
• Weight Normalization • https://arxiv.org/abs/1602.07868 
• Spectrum Normalization • https://arxiv.org/abs/1705.10941

# 【機器學習2021】09 卷積神經網路 (Convolutional Neural Networks, CNN)
https://youtu.be/OP5HcXJg2Aw?si=FuiWNu2ARWH7oCDT

CNN: 特化來影像處理的.
本課程用兩個觀點去理解CNN
Neuron Version Story: Each neuron only considers a receptive field. The neurons with different receptive fields share the parameters.
Filter Version Story: There are a set of filters detecting small patterns. Each filter convolves over the input image.
PS: Neuron Version Story中一組sharing parameter就是Filter Version Story中的一個filter!

Fully Connected Network: 每個Neuron都跟input的每一個dimension有一個weight(連接)
Observation: A neuron does not have to see the whole image (to identify a pattern).
Receptive field: 每個neuron只管一個有限的範圍(Receptive field). 不同neuron的Receptive field可以重複，可以全同. 每個neuron的Receptive field通常是連接的範圍, 如果符合你的需求也可以用不連接的範圍.
Typical Setting: 管色彩的3-channels總是全部包含. 自由度: kernel size(長寬), stride(平移量), padding(邊緣補值: 給處理到最邊緣的Receptive field使用). parameter sharing:  不同的Receptive field共享全同參數, 概念上代表在不同區域偵測相同的pattern, Two neurons with the same receptive field would not share parameters.

Fully Connected Layer -> 簡化自由度: Receptive Field + Parameter Sharing = Convolutional Layer.
CNN(Convolutional Layer Network) 

Pooling: 刪除偶數行的column跟奇數行的row, 用來把圖縮小(成1/4), 減少計算量, 但是會影響performance. Max/min/mean Pooling. 近年來如果計算能力夠, 也可以不用. optional.

Alpha Go: 使用CNN, 但沒有使用Pooling! 因為Pooling不適合圍棋.

 CNN is not invariant to scaling and rotation (we need data augmentation ☺)! CNN的架構, 不支援放大縮小旋轉(但平移可).  Why? 有空多想想.

 To learn more: 
Spatial Transformer Layer:  invariant to scaling and rotation.
https://www.youtube.com/watch?v=SoCywZ1hZak


# 【機器學習2021】10 # 自注意力機制 (Self-attention) (上)
https://youtu.be/hYdO9CscNes?si=W0UyrZPFuC5nw3NH

One-hot Encoding
![[Pasted image 20250914223213.png]]

Word Embedding
![[Pasted image 20250914223230.png]]
 To learn more: **Word Embedding** https://youtu.be/X7PH3NuYW0Q


• Each vector has a label. (focus of this lecture)
• The whole sequence has a label. (HW4)
• Model decides the number of labels itself. (HW5)


法1: 最陽春, 無法解決 "I saw a saw" 的label詞性問題.
![[Pasted image 20250914225615.png]]

法2: window. 你不容易知道window大小足夠大是說多大. 如果window設為整個sequence計算量太大, 且容易overfitting.
![[Pasted image 20250914225842.png]]

法3: Self-attention. 先從整個sequence of vector, 算出一樣大小的sequence of vector. 這樣可以看過整個sequence資訊.
![[Pasted image 20250914230112.png]]
計算細節:
老師說也不一定要用Dot-product, 可以用其他選擇. 課堂上舉例了另一種Additve選擇. (不過我覺得要比較兩個向量的相似程度, 還是用Dot比較直覺.)
![[Pasted image 20250914230624.png]]
![[Pasted image 20250914230610.png]]
![[Pasted image 20250914230823.png]]

師講解: 因為$\alpha$是機率向量, 最後b的值會趨近機率dominant對應的那個$v_i$分量.
Q: 最後從v算b的意義是什麼? 到底是在幹嘛?
# 【機器學習2021】11 # 自注意力機制 (Self-attention) (下)
https://youtu.be/gmsMY5kc-zw?si=NhomTXgXqPXk6kXe

Multi-head Self-attention: 為了算$q^{q,1}$, $q^{q,2}$, 除了原本的$W^q$還多了$W^{q,1}$, $W^{q,2}$(舉2head為例)
![[Pasted image 20250915011136.png]]


Q: 為什麼不乾脆改成下面這種架構, 這樣算$q^{q,1}$, $q^{q,2}$, 就只需要$W^{q,1}$, $W^{q,2}$. 這樣不是比較簡潔, 而且model跟上面的架構是等價的不是嗎?
![[Pasted image 20250915011644.png]]


Positional Encoding: 為了增加位置資訊, 多加了一個hand-crafted positional vector $e^i$
![[Pasted image 20250915012151.png]]


Q: 如果是我的話, 會直觀的改在dot product, 把$q^i k^{i'}$魔改加入depend on 兩點距離的decay multiplier, 像是 $q^i k^{i'} e^{-|i-i'|}$. (直覺: 距離越遠關係越小.) 但這個方法只能引進相對距離的資訊, 不像Positional Encoding可以引進絕對距離的資訊.
TODO: 未來可以嘗試實踐我的上述做法, 看在某些問題上會不會變得比較好. 也可以把這個作法用在下面的"把Self-Attention用在影像處理"的問題試驗看看效果. (此時, decay depend on 兩個pixel在圖上的二維距離)

Truncated Self-attention: 處理像是語音這種vector數量很大的data, 考慮全部計算量會太大, 改成人為的切段只考慮局部.

Self-Attention GAN(https://arxiv.org/abs/1805.08318): 把Self-Attention用在影像處理, 每一個pixel當作一個長度3(RGB 那個color channel)的vector.

Self-Attention: 每一個pixel考慮跟其他所有pixel之間的相關資訊. 容易overfit, 但data量大時可以做得比CNN好.
CNN: 只考慮同一個receptive field裡的pixel的資訊. Good for less data.
=> CNN 是簡化版的 self-attention.
事實上, 可以證明CNN是Self-Attention的特例.
On the Relationship between Self-Attention and Convolutional Layers https://arxiv.org/abs/1911.03584

(hw4)  要用CNN還是Self-Attention呢? 老師提示要用conformer(Q: 這啥?), 既有用到CNN也有用到Self-Attention.


Self-attention v.s. RNN: 
RNN: 因為近來逐漸被Self-attention取代, 這門課就只在這裡簡短提一下跟Self-attention做比較, 未來不會再提.
缺點: 中間那個架構不能平行處理. 單向的RNN每個節點只能考慮之前節點的資訊, 不像Self-attention可以考慮整個sequence的資訊, 雖然雙向版本的RNN可以同時考慮前後, 但是舉極端例子, 最左邊節點的資訊要從最左的memory開始, 一層層的往右傳遞並且不失去資訊, 才能讓最右邊的端點獲得資訊. (Reflections: 這跟我上面的TODO裡想像的距離decay也有有雷同之處?)

![[Pasted image 20250915151822.png]]

To learn more: Self-attention 加上一些什麼就會變成 RNN. https://arxiv.org/abs/2006.16236

To learn more: RNN過去的上課教學. (因為老師已經講Self-attention Win! 所以建議暫時可以先不看)
https://www.youtube.com/watch?v=xCGidAeyS4M
https://www.youtube.com/watch?v=Jjy6ER0bHv8

Self-attention for Graph: 把CNN用在graph上面是某一種變形的GNN
Attention Matrix的資訊不需要用learn的得到, 可以直接採用edge提供的資訊, 每個節點只考慮有連接的點的相關資訊即可. (Reflection: 是不是可以依照連接深度去分層考慮, 讓連接距離不同的節點重要性可以不一樣? Ex: 考慮下圖點1, 有一個"1-層Attention Matrix": 考慮點5,6,8, 有另一個"2-層Attention Matrix": 多考慮點3,4,7...")
![[Pasted image 20250915153030.png]]

To learn more: GNN
https://www.youtube.com/watch?v=eybCCtNKwzA
https://www.youtube.com/watch?v=M9ht8vsVEw8

廣義的Transformer就是Self-attention: 很多人在講Transformer時就是在說Self-attention.
Self-attention有很多變形都會被命名成XX-former.
To learn more: (短期不用看, 有興趣再說.) Self-attention如何做得更快更好仍然尚待研究. Efficient Transformers: A Survey https://arxiv.org/abs/2009.06732

# 【機器學習2021】12 Transformer (上)
https://youtu.be/n9TlOhRjYoc?si=6Zt1DVhPFCv9xw7F