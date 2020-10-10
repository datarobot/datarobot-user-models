prediction_value <- NaN

init <- function(...) {
    #library(stringr)
    prediction_value <<- 1
}

load_model <- function(input_dir) {
    prediction_value <<- prediction_value + 1
    "dummy"
}

score_unstructured <- function(model, ...) {
    prediction_value <<- prediction_value + 1
    list(data=toString(prediction_value))
}