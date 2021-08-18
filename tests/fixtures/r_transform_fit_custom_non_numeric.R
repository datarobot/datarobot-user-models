library(stringi)

fit <- function(X, y, output_dir, class_order=NULL, row_weights=NULL, ...){}

makestr <- function(n){
    stri_rand_strings(1, n)
}

transform <- function(X, transformer, y=NULL, ...){
    # ignore the model and output string
    as.data.frame(sapply(X, makestr), stringAsFactors=FALSE)
}