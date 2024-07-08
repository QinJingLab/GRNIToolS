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

ennet = function (E      = matrix(rnorm(10000),100,100),
                  K      = matrix(0,nrow(E),ncol(E)),
                  Tf     = 1:ncol(E),
                  s_s    = 1,
                  s_f    = 0.3,
                  M      = 5000,
                  nu     = 0.001,
                  scale  = TRUE,
                  center = TRUE,
                  optimization.stage  = 2) {
  N = ncol(E)
  S = nrow(E)
  E = scale(E,scale=scale,center=center)
  V = matrix(0, N, N)
  V = foreach(i = 1:N, .inorder = TRUE, .combine = "cbind") %dopar% {
    predictedI  = i
    predictorsI = setdiff(Tf, predictedI)
    validE      = K[,predictedI] == 0
    
    model = ennet.train(X.train = E[validE,predictorsI],
                        Y.train = E[validE,predictedI],
                        M.train = M,
                        nu = nu,
                        s_s = s_s,
                        s_f = s_f)
    result = rep(0, N)
    result[predictorsI] = model$importance
    result
  }
  
  if (optimization.stage > 0) {
    # first stage of re-evaluation
    s  = apply(V,1,var)
    S1 = matrix(rep(s,N),N,N)
    V  = V * S1
  }
  
  if (optimization.stage > 1) {
    # second stage of re-evaluation
    ko.experiments = which(rowSums(K)==1 & apply(K,1,max)==1)
    if (length(ko.experiments)>1) {
      S2 = matrix(1,N,N)
      E.ko = E[ko.experiments,]
      for (tf in Tf) {
        for (target in 1:N) {
          avg.ko   = mean(E.ko[K[ko.experiments,tf]==1,target])
          avg.n.ko = mean(E.ko[K[ko.experiments,tf]==0,target])
          std.dev  = sd(E.ko[,target])
          if (std.dev>0) {
            S2[tf,target]  = abs(avg.ko-avg.n.ko)/std.dev
          }
        }
      }
      V = V * S2
    }}
  
  # prepare the final adjacency matrix
  colnames(V) = colnames(E)
  rownames(V) = colnames(E)
  return(V)
}
