init <- function(...) {
    library(readr)
    library(stringr)
    library(stringi)
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
    if (is.raw(kwargs$data)) {
        data_text <- stri_conv(kwargs$data, "utf8")
    } else {
        data_text <- kwargs$data
    }
    count <- str_count(data_text, " ") + 1
    if (!is.null(kwargs$ret_mode) && kwargs$ret_mode == "binary") {
        ret = list(data=numToRawBigEndian(count), "mimetype"="application/octet-stream")
    } else {
        ret = list(data=toString(count))
    }
    ret
}