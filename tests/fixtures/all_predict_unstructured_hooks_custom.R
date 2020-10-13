prediction_value <- NaN

init <- function(...) {
    #library(stringr)
    prediction_value <<- 1
}

load_model <- function(input_dir) {
    prediction_value <<- prediction_value + 1
    "dummy"
}

score_unstructured <- function(model, data, ...) {
    prediction_value <<- prediction_value + 1
    toString(prediction_value)
}