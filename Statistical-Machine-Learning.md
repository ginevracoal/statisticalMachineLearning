# Statistical Machine Learning

| **LEZIONI** | LUN          | MAR          | MER     | GIO          |
| ----------- | ------------ | ------------ | ------- | ------------ |
| 9 - 11      | alg design   |              | stat ML | alg design   |
| 11 - 13     | stat methods | stat methods | stat ML | stat ML      |
| 14 - 16     | big data     | big data     |         | stat ML      |
| 16 - 18     | big data     | big data     |         | stat methods |


Luca Bortolussi, email: *lbortolussi@units.it*
office 328
[moodle](https://moodle2.units.it/course/view.php?id=2935)

**exam**: final project with presentation (possibily on datasets coming from companies) in groups of 3-4 people.


----------

07/03/18


## Inference and estimation

→ slides *03_basics.pdf*

We have a given probability distribution $$p(x,y)$$ where $$x=(x_1,...,x_m)$$, $$y=(y_1,...,y_d)$$ and $$y_i\in\{0,1\}$$.
We may want to compute $$p(x)=\int p(x,y) dy$$ or $$p(x|y)=\frac{p(x,y)}{p(x)}$$. 
In this particular case $$p(x)=\sum_{y\in\mathcal{Y}} p(x,y)$$, $$\mathcal{Y}=\{0,1\}^d$$, $$|\mathcal{Y}|=2^d$$.
If $$d=1000$$ then $$2^d \approx 10^{900}$$ and this sum is impossible to compute.
Here comes **inference,** where we aim at simplifying this complex model.

In **estimation** we are given a set of observations $$D=\{(x_i,y_i)\}_{i=1,...,N}$$ and $$p_{\theta}(x|y)$$, and we want to find the best $$\theta$$ that exploits data, or the best $$p(\theta|D)$$. 

**Bayesian** case: estimation = inference.


## Basics of probability
- Probability measure
- Probability space
- Random variables
  - mass function (discrete)
  - density function (continuous)
- rules of probability
  - normalization
  - sum rule
  - product rule
  - Bayes theorem
- independence
- inference: a form of reasoning about the model you are describing
- expectation
  - mean
  - variance
- covariance and correlation


## Some probability distributions


- Bernoulli 
- multinomial
- Poisson
- exponential
- **uniform**
- gamma
- beta
- Dirichlet
- **continuous gaussian/normal**
- **multivariate normal**


## Properties of multivariate normal

**Completing the square** 
*→ Bishop pag.78*

In the **one-dimensional case** we have:

                $$p(x|\mu, \sigma^2) = \frac{1}{\sqrt(2\pi\sigma^2)}exp(-\frac{(x-\mu)^2}{2\sigma^2})$$
                $$log \; p(x|\mu, \sigma^2) = const \; (-\frac{(x-\mu)^2}{2\sigma^2})$$
****
which is in the form

         $$log \; p(x) = const + bx -\frac{1}{2} a x^2$$

in this case we complete the square and get 

        $$log\; p(x)= -\frac{1}{2} a [x^2 - 2\frac{b}{a}x]=-\frac{1}{2} a [x^2 - 2\frac{b}{a}x+\frac{b^2}{a^2}]-\frac{ab^2}{2a^2}$$


                       $$= -\frac{1}{2a^{-1}} (x-a^{-1}b)^2 +...$$
                

which is a gaussian with $$\mu=a^{-1}b$$ and $$\sigma=a^{-1}$$.
In the **n-dimensional case** we can rewrite this as 

         $$log\; p(x) = -\frac{1}{2} x^T A x -b^T x + const=...= \frac{1}{2}[(x-A^{-1}b)^T A (x-A^{-1}b)]-\frac{1}{2}b^T A^{-1}b$$

gaussian with $$\Sigma=A^{-1}$$ and $$\mu = A^{-1}b$$.

**Linear transformation**
Gaussian distributions are closed for addition and linear combination


----------

08/03/18

*01_pandas_tutorial.ipynb*

[tutorial pivot table](http://pbpython.com/pandas-pivot-table-explained.html)
**

----------

14/03/18


## Basics

→ *03_basics.pdf*



----------

15/03/18

Expected loss
Bias-variance decomposition

**Information theory**
→ *p.60 Bishop*
$$-log(p)$$ is the self informatin of the event with probability $$p$$.

**Relative entropy** is a method for measuring the distance between two probability distributions.
Kullback Leiber has a unique minimum and reaches it only for the same distribution ($$p=q$$).


## Bayesian linear regression

→ *p.152 Bishop*
→ *04_bayesian_regression.pdf*

**Linear regression models**

- Regularized maximum likelihood
  *→ p.138 Bishop*
  - perché si usa l’inverse variance $$\beta^{-1}$$? si chiama **precision**
    precision is more "intuitive" than variance because it says *how concentrated* are the values around the mean rather than how much spread they are
    The more spread are the values around the mean (high variance) the less precise is they are (small precision). The smaller the variance, the greater the precision.
- train, validation and test data
- expected loss
- bias-variance decomposition

**Bayesian linear regression**

- posterior distribution
- marginal likelihood
- optimizing marginal likelihood
- Bayesian model comparison

**Dual representation and kernels**

- dual representation
  - $$a_i$$ are called duals because they uniquely identify $$w$$
- dual regression problem
- the kernel trick




----------

28/03/18

## (Bayesian) Linear classification

→ *04_bayesian_regression.pdf*

**Linear classifiers**

- discriminant function
  examples of discriminant function are SVM, classifcation trees…
- generative approach
  like Naive bayes classifier
- discriminative approach
- multiclass strategies

**Logistic regression**

- logit and probit regression
- numerical optimization
  - gradient descent
  - stochastic descent
- overfitting
- newton-rapson method
- multiclass LR
  - softmax function

**Laplace approximation**

- 1 dimension
- n dimension
  here A is an hessian…
- model comparison
- BIC

**Bayesian logistic regression**

- Laplace approximation of the posterior
- Predictive distribution
  - a linear combination of gaussians is still a gaussian
- probit approximation
- maximum likelihood vs bayesian approach


## Gaussian processes

→ *05_gaussian_processes.pdf*

**Random functions and bayesian regression**

- random functions
- exercise on marginal distributions
  - one point MD: $$f(x)$$ is gaussian with mean 0 and variance $$\phi^T(x)\phi(x)$$
  (guardare nelle slides…)
  - two point MG: $$(f(x_1),f(x_2))$$ is gaussian with mean 0 and covariance matrix as showd in the slides… 
- the gram matrix
- limitation of bayesian regression

**Gaussian processes**
they are practical because they give an analytical solution based on the inversion of a matrix

- example
  - la media influenza poco il problema, quello che conta è la varianza
  - nel plot ci sono vari campionamenti della distribuzione

dubbi:

[ ] studiare gaussian processes dal williams
[ ] stochastic process
----------

11 aprile


- noise-free prediction
  - joint prior distribution
- example: we observe the values on 5 points
- noisy predictions
  - noise has no covariance, so the second term $$\sigma^2 I$$ only depends on variance
- linear prediction
- posterior GP


[ ] fare i calcoli della covarianza

**Kernel functions**

- [integral transform](https://en.wikipedia.org/wiki/Integral_transform)
- eigenfunctions
- reproducing kernel Hilbert space
  - [Mercer theorem](https://en.wikipedia.org/wiki/Mercer%27s_theorem#Mercer.27s_condition)
- classification of kernel functions:
  - Gaussian kernel
  - Matérn kernel
  - exponential kernel
  - polynomial kernel
  - composition of kernels

**Hyperparameters**
*→ p.311 Bishop*

- marginal likelihood
  - in the graphs (b) is overfitting and (c) is underfitting
  - marginal likelihood is not a convex function so you could have more than a maximum
- non-constant prior mean
[](https://en.wikipedia.org/wiki/Integral_transform)
**GP classification**

- how to use logistic/probit regression
  - latent function
- Laplace approximation (as for LogR)
- expectation propagation
- pitfalls


## Scikit learn

→ *06_scikit_learn.ipynb*


- preprocessing
- linear fit

dubbi:

[ ] lebesgue integral
[ ] come si passa da kernel a hilbert space e viceversa tramite le eigenfunctions
[ ] sample from an observation
[ ] preprocessing del test set
[ ] methods python


----------

12 aprile

continuazione Scikit learn


- cross validation
  - Optimizing hyperparameters with cross-validation
  - 

