init <- function(...) {
    library(brnn)
    library(glmnet)
    library(dplyr)
}

transform <- function(data, model) {
    data[is.na(data)] <- 0
    if ("class" %in% names(data)) {
        data <- select(data, -class)
    }
    data
}