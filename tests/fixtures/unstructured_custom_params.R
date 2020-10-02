load_model <- function(input_dir) {
    "dummy"
}

score_unstructured <- function(model, ...) {
    kwargs <- list(...)
    ret <- list()

    if (!is.null(kwargs$ret_mimetype)) {
        ret <- append(ret, list(mimetype=kwargs$ret_mimetype))
    }
    if (!is.null(kwargs$ret_charset)) {
        ret <- append(ret, list(charset=kwargs$ret_charset))
    }
    ret <- append(ret, list(data=kwargs$ret_text))
    ret
}