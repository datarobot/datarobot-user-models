fit <- function(X, y, output_dir, class_order=NULL, row_weights=NULL, ...){}

transform <- function(X, transformer, y=NULL, ...){
    # ignore the model and output string
    as.data.frame(sapply(X, as.character))
}