init <- function(...) {
    library(readr)
    library(stringr)
}

load_model <- function(input_dir) {
    "dummy"
}

read_input_data <- function(input_filename) {
    data <- read_file(input_filename)
    data
}

score <- function(data, model, ...) {
    count <- str_count(data, " ") + 1
    ret = toString(count)
    ret
}