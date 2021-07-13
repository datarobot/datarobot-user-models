init <- function(...) {
    library(readr)
    library(stringr)
    library(pack)
    library(stringi)
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

score_unstructured <- function(model, data, query, ...) {
    kwargs <- list(...)

    print(c("Model: ", model))
    print(c("Incoming content type params: ", kwargs))
    print(c("Incoming data type: ", typeof(data)))
    print(paste0("Incoming data: ", paste0(data, collapse=" ")))

    print(c("Incoming query params: ", query))


    if (is.raw(data)) {
        data_text <- stri_conv(data, "utf8")
    } else {
        data_text <- data
    }
    count <- str_count(data_text, " ") + 1
    if (!is.null(query) && !is.null(query$ret_mode) && query$ret_mode == "binary") {
        ret_data = numToRawBigEndian(count)
        ret_kwargs = list(mimetype="application/octet-stream")
        ret = list(ret_data, ret_kwargs)
    } else {
        ret = toString(count)
    }
    ret
}