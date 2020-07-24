
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
            load_all(file.path(dd, pkgName))
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
