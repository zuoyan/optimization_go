Optimization framework in golang
================================

WIP: Not finished yet, but it shold work now.


Problem
-------

I have a dream,that, one day, anything can be optimized.

Let's start from one by one.

This's a small project with golang.

TODO: Rewrite in golang, instead of wrapper of C++, which one is better?

I put problems/notes first, so most people will read it before the document.

Document
--------

This framework have implemented some batch optimization solver, like gradient descent, conjugate gradient, lm bfgs.

We call the parameter space Point, this's used less than vector. A Point is just a scalar with PointHolder. The PointHolder can be anything which implements the PointHolderInterface. In this way, we can run distribute optimization with huge parameter spaces.

This framework has nothing with distribution, but it's designed and believed it's easy to use with big data and big parameter spaces. I have use it with big data, but haven't try it with big parameter spaces.
