# Copyright 2021 DataRobot, Inc. and its affiliates.
#
# All rights reserved.
#
# 
#
# This is proprietary source code of DataRobot, Inc. and its affiliates.
#
# Released under the terms of DataRobot Tool and Utility Agreement.
load_model <- function(input_dir) {
    "dummy"
}

score_unstructured <- function(model, data, ...) {
    ret_kwargs <- list()

    kwargs <- list(...)
    query <- kwargs$query
    if (query$ret_one_or_two == "one") {
        return(data)
    } else if (query$ret_one_or_two == "one-with-none") {
        return(list(data, NULL))
    } else {
        if (!is.null(query$ret_mimetype)) {
            ret_kwargs <- append(ret_kwargs, list(mimetype=query$ret_mimetype))
        }
        if (!is.null(query$ret_charset)) {
            ret_kwargs <- append(ret_kwargs, list(charset=query$ret_charset))
        }
        list(data, ret_kwargs)
    }
}