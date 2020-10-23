# Needed libraries
library(caret)
library(devtools)

init_hook <- FALSE
fit_hook <- FALSE

set.seed(1)

str_right <- function(string, n) {
  substr(string, nchar(string) - (n - 1), nchar(string))
}

get_nested_path <- function(path, file_pattern){
  if(str_right(path, 1) == '/'){
    path = subbstr(path, 1, nchar(path)-1)
  }
  return(
    list.files(path=path, pattern=file_pattern, recursive = TRUE, full.names=TRUE)
  )
}

init <- function(code_dir) {
    custom_path <- get_nested_path(code_dir, 'custom.[Rr]')
    if(length(custom_path) >= 1){
        custom_loaded <- import(custom_path[0])
    } else{
        stop('No custom file found.')
    }
  if (isTRUE(custom_loaded)) {
    init_hook <<- getHookMethod("init")
    fit_hook <<- getHookMethod("fit")
  }

  if (!isFALSE(init_hook)) {
    init_hook(code_dir=code_dir)
  }
}


load_data <- function(input_filename){
    tmp = readChar(input_filename, file.info(input_filename)$size)
    read.csv(text=gsub("\r","", tmp, fixed=TRUE), na.strings = c("NA", ""))
}


process_data <- function(input_filename, target_filename, target_name, num_rows){
    # read X
    df <- load_data(input_filename)
    # set num_rows
    if (num_rows == 'ALL'){
        num_rows <- nrow(df)
    } else {
        num_rows <- as.integer(num_rows)
    }
    # handle target
    if (!is.null(target_filename) || !is.null(target_name)){
        if (!is.null(target_filename)){
            y_unsampled <- load_data(target_filename)
            stopifnot(length(colnames(y_unsampled)) == 1)
            stopifnot(nrow(df) == nrow(y_unsampled))
            df <- cbind(df, y_unsampled)
            target_name <- colnames(y_unsampled)
        }
    df <- df[!(is.na(df[target_name])), ]
    X <- df[,!(names(df) %in% c(target_name))]
    X <- X[sample(nrow(X), size=num_rows, replace=TRUE ), ]

    y <- df[,target_name, drop=FALSE]
    y <- y[sample(nrow(y), size=num_rows, replace=TRUE),]

    } else {

        y <- NULL
        X <- df[sample(nrow(df), size=num_rows, replace=TRUE ), ]

    }

    return (list('X' = X, 'y' = y, 'num_rows' = num_rows))

}


process_weights <- function(X, weights_filename, weights, num_rows){
  if (!is.null(weights_filename)){
    row_weights <- load_data(weights_filename)
    row_weights <- row_weights[sample(nrow(row_weights), size=num_rows, replace=TRUE ), ]
  } else if(!is.null(weights)){
    if (! weights %in% colnames(X)){
      stop( paste("The column name",
                  weights,
                  "is not one of the columns in your training data", sep=' '))
    }
    row_weights <- X[, weights, drop=FALSE]
  } else {
    row_weights = NULL
  }
  row_weights
}

#' Fits and saves a model using user-provided Fit method
#'
#' @param data data.frame from which to train model
#' @param output_dir directory to save model
#' @param class_order array containing [negative label, positive label] or NULL
#' @param row_weights array with row weights, or NULL
#'

outer_fit <- function(output_dir, input_filename, target_filename,
                      target_name, num_rows, weights_filename, weights,
                      positive_class_label, negative_class_label) {

    processed_data <- process_data(input_filename, target_filename, target_name, num_rows)

    X <- processed_data$X
    y <- processed_data$y
    num_rows <- processed_data$num_rows

    row_weights <- process_weights(X, weights_filename, weights, num_rows )

    if (!is.null(positive_class_label) && !is.null(negative_class_label)){
        class_order <- c(negative_class_label, positive_class_label)
    } else {
        class_order <- NULL
    }

    if (!isFALSE(fit_hook)) {
        kwargs <- list()
        kwargs <- append(kwargs, list(X=X,
                                      output_dir=output_dir,
                                      row_weights=row_weights))
        if (!is.null(y)) {
            kwargs <- append(kwargs, list(y=y,
                                            class_order=class_order))
        }

        do.call(fit_hook, kwargs)
        } else {
            stop(sprintf("No Fit method provided."))
        }
}
