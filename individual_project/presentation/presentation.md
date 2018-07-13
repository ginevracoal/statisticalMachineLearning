---
title: Kernel approximations on large-scale problems
subtitles: Nyström method and Random Fourier features
author: Ginevra Carbone
date: "*25 Luglio 2018*"
output:
  ioslides_presentation:
  fig_width: 7
  incremental: yes
  smaller: yes
  widescreen: yes
html_document:
  toc: yes
editor_options:
  chunk_output_type: inline
always_allow_html: yes
---
          
## Introduction 

Kernel methods are based on the idea of projecting data points into a high-dimensional **feature space** and searching for the optimal **separating hyperplane** in that feature space.

<!-- ![](figures/kernel_trick.png){ width=20 } -->

<div align="center">
<img src="figures/kernel_trick.png" width=600>
</div>

---

## Introduction

The main limitation of these methods is their high **computational cost**, which is at least quadratic in the number of training points, due the calculation of the kernel matrix.

Low-rank decompositions of the matrix (like **Cholesky decomposition**) are less expensive, but they still require to compute the kernel matrix.

<!-- dire qualcosa di più su cholesky.. -->

## Introduction

How can we avoid computing kernel matrix? 

We can simply **approximate** the matrix or directly the kernel function.

Let's explore two methods based on low-rank matrix approximations: **Nyström method** and **random Fourier features**.

## Notation

<!-- prima descrivo i metodi, poi vedo cosa mi serve definire... in caso scrivo un'appendice -->

Let $k:\mathcal{X}\times\mathcal{X}\rightarrow \mathbb{R}$ be a kernel ^[1], $\mathcal{H}_k$ the corresponding Reproducing kernel Hilbert space, $n$


[^1]: $\mathcal{X}\in \mathbb{R}^d$

## Nyström method

<!-- These are the main steps:  -->

Take $m$ random samples with repetition $\hat{x}_1,\dots, \hat{x}_m$ of training data and consider the corresponding kernel matrix $\hat{K} = [k(\hat{x_i},\hat{x_j})]_{m\times m}$.

Supposing that $\hat{K}$ has rank $r$, the main idea of the algorithm is to **approximate $K$** by the low-rank matrix 
$$
{\hat{K}}_r = K_b \hat{K}^+ {K_b}^{-1}
$$
where $K_b=[k(x_i,\hat{x_j})]_{N\times m}$ and $\hat{K}^+$ is the Moore-Penrose pseudo inverse of $\hat{K}$ [1],[2].

<!-- dire che la matrice ha rango r -->

## Random Fourier features

## Theoretical differences

- data dependence
- computational complexity


## Testing phase

- mnist
- adult
- forest
- leaf
- ...

## Empirical results

- regression vs classification performances

## Considerations


## References

[1] P. Drineas and M. W. Mahoney, "On the Nystrom Method for Approximating a Gram Matrix for Improved Kernel-Based Learning", 2005, JMLR,
http://www.jmlr.org/papers/volume6/drineas05a/drineas05a.pdf.

[2] T. Yang, Y.-F. Li, M. Mahdavi, R. Jin and Z.-H. ZhouNystrom, "Method vs Random Fourier Features: A Theoretical and Empirical Comparison", 2012, NIPS, https://papers.nips.cc/paper/4588-nystrom-method-vs-random-fourier-features-a-theoretical-and-empirical-comparison.pdf.

[3] A. Rahimi and B. Recht, "Random features for large-scale kernel machines" , 2007, NIPS, https://www.robots.ox.ac.uk/~vgg/rg/papers/randomfeatures.pdf.









