install.packages('devtools')
devtools::install_github("rstudio/renv")
library(renv)
renv::settings$snapshot.type("all")
renv::init()

install.packages(c('DatabaseConnector', 'dplyr', 'knitr', 'rJava', 'RJDBC'), quiet = TRUE)
install.packages(c('IRkernel', 'devtools', 'mplot','readr','forestplot','survminer','SqlRender','sqldf','r2d3','tidyverse','survMisc','jsonlite','DBI','ggplot2','googleVis','htmlwidgets','lubridate','magrittr','plotly','Rcpp','reshape','reshape2','rgl','stringr','tibble','tidyr','vcd','viridis','XML','xts'), quiet = TRUE)
renv::install("snowflakedb/dplyr-snowflakedb")
# You will need to create a GitHub PAT, authorize it to the datarobot organization
# and set it to the env var GITHUB_PAT, ideally in .Renviron file (`usethis::edit_r_environ()`)
remotes::install_github("datarobot/rsdk@v2.31.0.9000", subdir = "datarobot.apicore")
remotes::install_github("datarobot/rsdk@v2.31.0.9000", subdir = "datarobot")
# If instead you want to just install the latest version every time, use these:
# Not recommended since this will lead to reproducibility issues
# install_github("datarobot/rsdk", subdir = "datarobot.apicore", ref = github_release())
# install_github("datarobot/rsdk", subdir = "datarobot", ref = github_release())
library(datarobot)

renv::snapshot()
