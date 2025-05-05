# ece473-homework-3-solved
**TO GET THIS SOLUTION VISIT:** [ECE473 Homework 3 Solved](https://www.ankitcodinghub.com/product/ece473-homework-3-solved/)


---

📩 **If you need this solution or have special requests:** **Email:** ankitcoding@gmail.com  
📱 **WhatsApp:** +1 419 877 7882  
📄 **Get a quote instantly using this form:** [Ask Homework Questions](https://www.ankitcodinghub.com/services/ask-homework-questions/)

*We deliver fast, professional, and affordable academic help.*

---

<h2>Description</h2>



<div class="kk-star-ratings kksr-auto kksr-align-center kksr-valign-top" data-payload="{&quot;align&quot;:&quot;center&quot;,&quot;id&quot;:&quot;94482&quot;,&quot;slug&quot;:&quot;default&quot;,&quot;valign&quot;:&quot;top&quot;,&quot;ignore&quot;:&quot;&quot;,&quot;reference&quot;:&quot;auto&quot;,&quot;class&quot;:&quot;&quot;,&quot;count&quot;:&quot;0&quot;,&quot;legendonly&quot;:&quot;&quot;,&quot;readonly&quot;:&quot;&quot;,&quot;score&quot;:&quot;0&quot;,&quot;starsonly&quot;:&quot;&quot;,&quot;best&quot;:&quot;5&quot;,&quot;gap&quot;:&quot;4&quot;,&quot;greet&quot;:&quot;Rate this product&quot;,&quot;legend&quot;:&quot;0\/5 - (0 votes)&quot;,&quot;size&quot;:&quot;24&quot;,&quot;title&quot;:&quot;ECE473 Homework 3 Solved&quot;,&quot;width&quot;:&quot;0&quot;,&quot;_legend&quot;:&quot;{score}\/{best} - ({count} {votes})&quot;,&quot;font_factor&quot;:&quot;1.25&quot;}">

<div class="kksr-stars">

<div class="kksr-stars-inactive">
            <div class="kksr-star" data-star="1" style="padding-right: 4px">


<div class="kksr-icon" style="width: 24px; height: 24px;"></div>
        </div>
            <div class="kksr-star" data-star="2" style="padding-right: 4px">


<div class="kksr-icon" style="width: 24px; height: 24px;"></div>
        </div>
            <div class="kksr-star" data-star="3" style="padding-right: 4px">


<div class="kksr-icon" style="width: 24px; height: 24px;"></div>
        </div>
            <div class="kksr-star" data-star="4" style="padding-right: 4px">


<div class="kksr-icon" style="width: 24px; height: 24px;"></div>
        </div>
            <div class="kksr-star" data-star="5" style="padding-right: 4px">


<div class="kksr-icon" style="width: 24px; height: 24px;"></div>
        </div>
    </div>

<div class="kksr-stars-active" style="width: 0px;">
            <div class="kksr-star" style="padding-right: 4px">


<div class="kksr-icon" style="width: 24px; height: 24px;"></div>
        </div>
            <div class="kksr-star" style="padding-right: 4px">


<div class="kksr-icon" style="width: 24px; height: 24px;"></div>
        </div>
            <div class="kksr-star" style="padding-right: 4px">


<div class="kksr-icon" style="width: 24px; height: 24px;"></div>
        </div>
            <div class="kksr-star" style="padding-right: 4px">


<div class="kksr-icon" style="width: 24px; height: 24px;"></div>
        </div>
            <div class="kksr-star" style="padding-right: 4px">


<div class="kksr-icon" style="width: 24px; height: 24px;"></div>
        </div>
    </div>
</div>


<div class="kksr-legend" style="font-size: 19.2px;">
            <span class="kksr-muted">Rate this product</span>
    </div>
    </div>
<div class="page" title="Page 2">
<div class="layoutArea">
<div class="column">
Problem 1: Gaussian Naive Bayes Classifier

Calculation of Bayes Theorem can be simplified by making some assumptions, such as independence between all input variables. Although a dramatic and unrealistic assumption, this has the effect of making the calculations of the conditional probability tractable and results in an effective classification model referred to as Naive Bayes.

The Naive Bayes algorithm has proven effective and therefore is popular for text classification tasks. The words in a document may be encoded as binary (word present), or count (word occurrence) input vectors and binary, multinomial, or Gaussian probability distributions used.

In this problem, you will implement Gaussian Naive Bayes Classifier and test it on two datasets: 1. breast cancer dataset

2. digits dataset

Digits dataset consist of 1797 samples with 10 classes. Each sample has 64 features, which is originally 8×8 image. Your task for both dataset is to predict correct class for the test samples.

(Note: When you compute Bayes rule, denominator cannot be 0. In other words, a value inside log cannot be equal or less than 0.)

Tasks in Problem 1:

1. Implement function Naive Bayes.mean

2. Implement function Naive Bayes.std

3. Implement function Naive Bayes.gen by class

4. Implement function Naive Bayes.calc gaussian dist 5. Implement function Naive Bayes.predict

</div>
</div>
<div class="layoutArea">
<div class="column">
2

</div>
</div>
</div>
<div class="page" title="Page 3">
<div class="layoutArea">
<div class="column">
Problem 2: Two-Layered Neural Network

Based on the structure of Logistic regression of Homework 2, you will extend the structure to two-layers neural network in this problem. Note that the counting index of “layers” starts with the first hidden layer up to the output layer. Thus, as illustrated in Fig 1, two-layered neural network has one hidden layer with a certain number of hidden units. The number of hidden units can vary considering various aspects such as how complex the input feature is or how deep the neural network is.

Figure 1: Illustration of two-layered neural network.

In this homework, you will implement a two-layered neural network with a hidden layer with 64 units and ReLU activation before fed forward to the output layer. Whereas for the output layer, you will use sigmoid function instead.

Denote x ∈ RD as an input data with D dimensions and y ∈ {0,1} as the corresponding true binary label. We denote f(x;θ) : RD → R as a two-layered neural network that we want to train. The goal is to find f that minimizes the loss L in Eq.(1).

L(f(x),y) = LBCE(f(x),y)+λ∥θ∥2 . (1)

This loss L consist of two terms: 1) binary cross entropy loss (LBCE) and 2) regularization. The first term LBCE is identical to the logistic regression in hw2 as in Eq.(2). The second term is the regularization with the weight λ which is for better generalization and to prevent the model from over-fitting to the training data. Note that θ includes all trainable parameters including weights and biases in all layers. To update the parameters with back propagation in the neural network, you will use mini-batch stochastic gradient descent method as in Homework 2.

􏰀􏰁

LBCE(f(x),y)=E(x,y) ylog(f(x))+(1−y)log(1−f(x)) . (2)

Note: hyperparameters are provided as follows. Do not change the hyperparameter setting. • learning rate = 10−3,

• batch size = 64,

• iterations number = 2000,

• regularization hyperparameter λ in Eq. (1) (i.e., reg in hw3 submission.py) = 10−3. In this problem, you will test the neural network on breast cancer dataset.

Tasks in Problem 2:

1. Implement function Neural Network.activation 2. Implement function Neural Network.sigmoid

3. Implement function Neural Network.loss

</div>
</div>
<div class="layoutArea">
<div class="column">
3

</div>
</div>
</div>
<div class="page" title="Page 4">
<div class="layoutArea">
<div class="column">
4. Implement function Neural Network.fit

5. Implement function Neural Network.predict

</div>
</div>
<div class="layoutArea">
<div class="column">
4

</div>
</div>
</div>
