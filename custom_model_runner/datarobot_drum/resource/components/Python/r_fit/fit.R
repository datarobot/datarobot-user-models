# Needed libraries
library(caret)
library(devtools)

init_hook <- FALSE
fit_hook <- FALSE

#' Import R source files as a named package
#'
#' @param srcFiles character, file paths to load as the package
#' @param pkgName character or NULL, name to give the package
#'
#' @return bool, TRUE if the package was succussfully loaded, FALSE otherwise
#' @export
#'
#' @examples import("~/Documents/R/custom.R", "myPackage")
import <- function(srcFiles, pkgName = "custom") {
    dd <- tempdir()
    on.exit(unlink(file.path(dd, pkgName), recursive=TRUE))
    tryCatch(
        {
            package.skeleton(name=pkgName, path = dd, code_files=srcFiles)
            load_all(file.path(dd, pkgName)) # warn_conflicts = FALSE
            return(TRUE)
        },
        error = function(cond) {
            message(c(cond, "\n"))
            return(FALSE)
        }
    )
}

#' Get a method from a package
#'
#' @param name character, the name of the method to retrieve
#' @param pkgName character or NULL, the package to look in for the method
#'
#' @return function if the method is found or FALSE
#' @export
#'
#' @examples getHookMethod("foo", "myPackage")
getHookMethod <- function(name, pkgName = "custom") {
    tryCatch(
        {
            hook = getExportedValue(pkgName, name)
            if (is.function(hook)) {
                return(hook)
            } else {
                message(name, " is not a method")
                return(FALSE)
            }
        },
        error = function(cond) {
            message(c(cond, "\n"))
            return(FALSE)
        }
    )
}

init <- function(code_dir) {
  if (file.exists(file.path(code_dir, 'custom.R'))){
    custom_path <- file.path(code_dir, "custom.R")
  } else {
    custom_path <- file.path(code_dir, "custom.r")
  }
  custom_loaded <- import(custom_path)
  if (isTRUE(custom_loaded)) {
    init_hook <<- getHookMethod("init")
    fit_hook <<- getHookMethod("fit")
  }

  if (!isFALSE(init_hook)) {
    init_hook(code_dir=code_dir)
  }
}


#' Fits and saves a model using user-provided Fit method
#'
#' @param data data.frame from which to train model
#' @param output_dir directory to save model
#' @param class_order array containing [negative label, positive label] or NULL
#' @param row_weights array with row weights, or NULL
#'

outer_fit <- function(X, y, output_dir, class_order=NULL, row_weights=NULL) {
    if (!isFALSE(fit_hook)) {
        kwargs <- list()
        kwargs <- append(kwargs, list(X=X,
                                      y=y,
                                      output_dir=output_dir,
                                      class_order=class_order,
                                      row_weights=row_weights))

        do.call(fit_hook, kwargs)
        } else {
            stop(sprintf("No Fit method provided."))
        }
}
