#    This is an implementation of ENNET algorithm for Gene Regulatory Network
#    inference from mRNA expression data, in form of an R package.
#    Copyright (C) 2013  Janusz Slawek

#    This program is free software: you can redistribute it and/or modify
#    it under the terms of the GNU General Public License as published by
#    the Free Software Foundation, either version 3 of the License, or
#    any later version.

#    This program is distributed in the hope that it will be useful,
#    but WITHOUT ANY WARRANTY; without even the implied warranty of
#    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#    GNU General Public License for more details.

#    You should have received a copy of the GNU General Public License
#    along with this program, see LICENSE.

ennet.test = function(model,X.test,Y.test,M.test) {
  if (!is.matrix(X.test)) {
    stop("Error: X.test must be N-by-P matrix")
  }
  N.test = nrow(X.test)
  P.test = ncol(X.test)
  if (!is.vector(Y.test) || length(as.vector(Y.test))!=N.test) {
    stop("Error: Y.test must be N-element vector.")
  }
  if (model$P.train != P.test) {
    stop("Error: dimensionality of training and test data must agree.")
  }
  if (!is.numeric(M.test) || as.integer(M.test)<=0) {
    stop("Error: M must be a number greater than 0.")
  }
  if (!is.numeric(M.test) || model$M.train < M.test) {
    stop("Error: number of iterations M must be lower or equal to model size.")
  }
  if (!is.numeric(model$f0)) {
    stop("Error: f0 of model is corrupted")
  }
  if (!is.vector(model$feature.split.index) || length(as.vector(model$feature.split.index)) != model$M.train) {
    stop("Error: feature.split.index of model is corrupted")
  }
  if (!is.vector(model$feature.split.thr) || length(as.vector(model$feature.split.thr)) != model$M.train) {
    stop("Error: feature.split.thr of model is corrupted")
  }
  if (!is.vector(model$gamma_l) || length(as.vector(model$gamma_l)) != model$M.train) {
    stop("Error: gamma_l of model is corrupted")
  }
  if (!is.vector(model$gamma_r) || length(as.vector(model$gamma_r)) != model$M.train) {
    stop("Error: gamma_r of model is corrupted")
  }
  if (!is.numeric(model$nu)) {
    stop("Error: nu of model is corrupted")
  }
  
  result = .C("test_regression_stump_R",
              as.integer(N.test),
              as.integer(P.test),
	      as.integer(model$P.train),
              as.double(X.test),
              as.double(Y.test),
              as.integer(M.test),
	      as.integer(model$M.train),
	      as.double(model$nu),
              as.double(model$f0),
              as.integer(model$feature.split.index-1),
              as.double(model$feature.split.thr),
              as.double(model$gamma_l),
              as.double(model$gamma_r),
              loss=as.double(rep(0.0,M.test)),
              max.M.prediction=as.double(rep(0.0,N.test)),
              PACKAGE="ennet"
  )
  
  prediction = list(
    loss=result$loss,
    max.M.prediction=result$max.M.prediction
  )
  class(prediction) = "ennet.prediction"
  
  return(prediction)
}
