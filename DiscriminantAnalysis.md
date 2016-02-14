### Discriminant Analysis

**Kedar Page** \
**14 Feb 2016**

*Short notes from the lectures by Trevor Hastie and Rob Tibshirani, the authors of the book "Elements of Statistical learning". These notes are for reference only and are not intended or suitable to be a tutorial on this topic.*


##### Introduction
* An approach to classification

* Model distribution of $X$ in classes separately and then use
Bayes theorem to find $P(Y|X)$

* Gaussian distributions lead to linear or quadratic discriminant analysis, but can be used for other distributions as well.

* Bayes theorem for classification

$$
P(Y = k | X = x) = \frac{P(X = x|Y=k) P(Y = k)}{P(X=k)}
$$

* Can be written as

$$
P(Y = k | X = x) = \frac{ \pi_{k} f_{k}(x)}
{\sum_{l=1}^{K} \pi_{l} f_{l}(x)}
$$

Where $f_{k}(x)$ is the density of $X$ in class $k$ and  $\pi_{k}$  is the prior probability for class $k$. We then classify based on the highest probability $P(Y = k | X = x)$.

* Discriminant analysis performs better than logistic regression when the class are separated better. Parameter estimates are not stable in logistic regression when classes are well separated.

##### LDA - Linear Discriminant Analysis

* If the density functions of each variable in $x$ can be assumed to be Gaussian with the same covariance matrix in each of the classes and with possible correlations with each other we'll have a multivariate Gaussian density function for $f(x)$

$$
f(x) = \frac{1}{(2\pi)^{p/2}|\Sigma|^{1/2}} e^{ - \frac{(x-\mu)^T |\Sigma^{-1}|(x-\mu)}{2}}
$$
Where $\Sigma$ is the covariance matrix of $X$.

* Taking log and simplifying gives us the discriminant function (or discriminant score) $\delta_k(x)$. Instead of the probability $P(Y = k| X = x)$, we classify based on  the highest value of the $\delta_k(x)$.

$$
\delta_k(x)  = x^T \Sigma^{-1}\mu_k
 - \frac{1}{2}\mu_k^T \Sigma^{-1}\mu_k + log(\pi_k)
$$


* $\delta_k(x)$ is a linear function in $x$. Classifying based on this discriminant will give us the *Bayes decision boundaries* between the classes which should result in the least misclassification error possible for any classifier.

* However since we often do not know the population parameters of our underlying distributions and make do with sample estimates, the resulting decision boundaries will be close to the Bayes decision boundaries but not an exact match.

* The estimates $\delta_k(x)$ can be used to get back the probabilities too.

$$
\hat{P}(Y = k | X = x) =
\frac{e^{\hat{\delta}_k(x)}}{\sum_{l=1}^K e^{\hat{\delta}_k(x)}}
$$

* Classifying to largest $\delta_k(x)$ is the same as classifying to the class with highest $\hat{P}(Y = k | X = x)$.

##### QDA - Quadratic Discriminant Analysis

* With Gaussians for $f_k(x)$ but different covariance matrices $\Sigma_k$ in each class we get *quadratic discriminant analysis*. The discriminant function is:

$$
\delta_k(x) = - \frac{1}{2}(x-\mu_k)^T \Sigma_k^{-1}(x-\mu_k) + log(\pi_k)
$$

* This gives parabolic Bayes decision boundaries instead of linear ones.

* The classification performance does not usually suffer very much if one chooses LDA instead of QDA since.  

##### Naive Bayes Classifer

* When we have a lot of features, it is computationally infeasible to compute the covariance matrices.

* If we make the very likely wrong assumption that the variables in $X$ are not correlated in each of the classes. i.e.

$$
f_k(x) = \Pi_{j=1}^p f_{jk}(x_j)
$$

This means that $\Sigma_k$ are all diagonal.

* This gives the discriminant function

$$
\delta_k(x) = -\frac{1}{2} \sum_{j=1}^p \frac{(x_j - \mu_{kj})^2}{\sigma_{kj}^2} + log(\pi_k)
$$

* This works for mixed type datasets as we can replace $f_{kj}(x_j)$ with the *probability-mass-functions (histograms)* over discrete categories.

* Despite these strong and very likely wrong assumption Naive Bayes classifier produces good results on large datasets.
