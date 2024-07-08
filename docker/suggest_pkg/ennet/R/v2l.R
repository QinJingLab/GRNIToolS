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

v2l = function(V,max=100000,check.names=TRUE) {
  V            = data.frame(V,check.names=check.names)
  N            = nrow(V)
  M            = ncol(V)
  targets      = colnames(V)
  tfs          = rownames(V)
  connList     = data.frame(matrix(0,N*M,3))
  
  connList[,1] = rep(tfs,M)
  connList[,2] = rep(targets,1,each=N)
  connList[,3] = as.vector(unlist(V))
  
  self = seq(from=1,to=(N*M),by=(N+1))
  
  connList     = connList[-self,]
  connList     = arrange(connList,X3,decreasing=TRUE)
  
  connList[,3] = connList[,3]/connList[1,3]
  if (nrow(connList)>max) {
    connList     = connList[1:max,]
  }
  return(connList)
}
