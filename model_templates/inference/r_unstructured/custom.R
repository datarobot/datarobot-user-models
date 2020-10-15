init <- function(...) {
    library(readr)
    library(stringr)
    library(pack)
}

load_model <- function(input_dir) {
    "dummy"
}

numToRawBigEndian <- function(num, nBytes = 4) {
    # pack's numToRaw returns bytes in platform's defined order
    numRaw <- numToRaw(num, nBytes)
    if (.Platform$endian == "little") {
        numRaw <- rev(numRaw)
    }
    numRaw
}

score_unstructured <- function(model, ...) {
    kwargs <- list(...)
    if (!is.null(kwargs$data)) {
        data_text <- stri_conv(kwargs$data, "utf8")
    } else {
        data_text <- kwargs$text
    }
    count <- str_count(data_text, " ") + 1
    if (kwargs$ret_mode == "binary") {
        ret = list(data=numToRawBigEndian(count))
    } else {
        ret = list(data=toString(count))
    }
    ret
}