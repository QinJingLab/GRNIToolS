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

ennet.train = function(X.train,Y.train,s_f=0.3,s_s=1,M.train=5000,nu=0.001) {
  if (!is.matrix(X.train)) {
    stop("Error: X must be N-by-P matrix.")
  }
  N.train = nrow(X.train)
  P.train = ncol(X.train)
  if (!is.vector(Y.train) || length(as.vector(Y.train))!=N.train) {
    stop("Error: Y must be N-element vector.")
  }
  if (!is.numeric(s_f) || as.double(s_f)<=0 || as.double(s_f)>1) {
    stop("Error: s_f must be greater than 0 and lower or equal than 1.")
  }
  if (!is.numeric(s_s) || as.double(s_f)<=0) {
    stop("Error: s_s must be greater than 0.")
  }
  if (!is.numeric(M.train) || as.integer(M.train)<=0) {
    stop("Error: M must be greater than 0.")
  }
  if (!is.numeric(nu) || as.double(nu)<=0 || as.double(nu)>1) {
    stop("Error: nu must be greater than 0 and lower or equal than 1.")
  }
  
  result = .C("train_regression_stump_R",
              as.integer(N.train),
              P.train=as.integer(P.train),
              as.double(X.train),
              as.double(Y.train),
              as.double(s_f),
              as.double(s_s),
              M.train=as.integer(M.train),
              nu=as.double(nu),
              importance=as.double(rep(0.0,P.train)),
              f0=as.double(0),
              feature.split.index=as.integer(rep(0,M.train)),
              feature.split.thr=as.double(rep(0.0,M.train)),
              gamma_l=as.double(rep(0.0,M.train)),
              gamma_r=as.double(rep(0.0,M.train)),
              PACKAGE="ennet"
  )
  
  model = list(
    P.train=result$P.train,
    M.train=result$M.train,
    nu=result$nu,
    importance=result$importance,
    f0=result$f0,
    feature.split.index=result$feature.split.index+1,
    feature.split.thr=result$feature.split.thr,
    gamma_l=result$gamma_l,
    gamma_r=result$gamma_r
  )
  class(model) = "ennet.model"
  
  return(model)
}

