$$\newcommand{\X}{\mathcal{X}}$$
$$\newcommand{\Y}{\mathcal{Y}}$$
$$\newcommand{\H}{\mathcal{H}}$$
$$\newcommand{\E}{\mathbb{E}}$$

#Counterfactual Risk Minimization

This paper tackles the bandit problem given logged data (not online). The data is logged using some baseline policy.

## Setup

The input is $$x\in \X$$ and the prediction is $$y\in \Y$$. A hypothesis $$h(\Y|x)\in \H$$ defines a probability distribution and for each $$x$$, samples from $$h(\Y|x)$$ which is often shortened to $$h(x)$$. 

We only observe feedbacks $$\delta(x,y)$$ for $$y$$ sampled from $$h(x)$$. Small values for $$\delta(x,y)$$ indicate satisfaction for user with attributes $$x$$ with prediction $$y$$. The expected loss of a hypothesis $$R(h)$$ is 

$$R(h) = \E_{x\sim Pr(\X)}\E_{y\sim h(x)}[\delta(x,y)] = \E_h[\delta(x,y)]$$

The data is logged according to a stationary policy $$h_0(x)$$. The data collected from this system comes in triples $$\{ (x_i,y_i,\delta_i)\}_{i=1}^n$$ where $$y_i \sim h_0(x_i)$$. 

## Importance Sampling

Using importance sampling, one has 

$$R(h) = \E_{h_0}\left[ \delta(x,y) \frac{h(y|x)}{h_0(y|x)}\right]$$ For the observed data, we can define propensity scores $$ p_i \equiv h_0(y_i|x_i)$$ which are known. The MC estimate of $$R(h)$$ is given by 

$$\hat{R}(h) = \frac{1}{n} \sum_{i=1}^n \delta_i \frac{h(y_i|x_i)}{p_i}$$

Directly estimating $$\hat{R}(h)$$ has some problems. 
* It is not invariant to additive transformations. We can first assume $$\delta(x,y) \in [-1,0]$$. 
* The estimator can have unbounded variance which means the estimator $$\hat{R}(h)$$ can be arbitrarily far away from $$R(h)$$.

Previous work has studied the clipped risk and estimator. 

$$ R^M(h) = \E\left[ \delta(x,y) \min \left\{ M, \frac{h(y|x)}{h_0(y|x)}\right\}\right]$$

$$ \hat{R}^M(h) = \frac{1}{n} \sum_{i=1}^n \delta_i \min \left\{ M, \frac{h(y_i|x_i)}{p_i}\right\}$$

The Inverse Propensity scoring training objective tries to minimize the clipped estimator. 

$$\hat{h}^{IPS} = \underset{h \in \mathcal{H}}{\mathrm{argmin}}\left\{ \hat{R}^M(h)\right\}$$

There are some issues with this still as it is variance sensitive for different hypotheses depending on how well they match . 

## Generalization Error Bound

For each $$\mathcal{H}$$, we define a corresponding deterministic function class $$\mathcal{F}_\mathcal{H}= \{ f_h: \X \times \Y \mapsto [0,1]\}$$. Each $$h\in \mathcal{H}$$ corresponds to a function $$f_h \in \mathcal{F}_\mathcal{H}$$ ,

$$f_h(x,y) = 1 + \frac{\delta(x,y)}{M} \min \left\{ M, \frac{h(y|x)}{h_0(y|x)}\right\}$$

The goal is to use covering numbers for $$\mathcal{F}_\mathcal{H}$$ .

Lemma: $$ \E_{h_0}[f_h(x,y)] = 1 + \frac{R^M(h)}{M}$$

Definition: The covering number is $$ \mathcal{N}_\infty(\epsilon,\mathcal{F},n) = \sup_{(x_i,y_i) \in (\X \times \Y)^n} \mathcal{N}(\epsilon, \mathcal{F}(\{ (x_i,y_i)\}, \| \cdot\|_\infty)$$where $$ \mathcal{F}(\{ (x_i,y_i)\} )$$ is the function class $$\mathcal{F}$$ conditioned on a sample $$\{ (x_i,y_i)\} $$  $$\mathcal{F}(\{ (x_i,y_i)\} ) = \{ (f(x_1,y_1),\ldots,f(x_n,y_n):f\in \mathcal{F}\}.$$

To compactify notation, the 