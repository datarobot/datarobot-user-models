init <- function(...) {
    library(brnn)
    library(glmnet)
}

transform <- function(data, model) {
    data[is.na(data)] <- 0
    data
}