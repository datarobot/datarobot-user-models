install.packages('devtools')
devtools::install_github("rstudio/renv")
library(renv)
renv::settings$snapshot.type("all")
renv::init()

install.packages(c('DatabaseConnector', 'dplyr', 'knitr', 'rJava', 'RJDBC'), quiet = TRUE)
install.packages(c('IRkernel', 'devtools', 'mplot','readr','forestplot','survminer','SqlRender','sqldf','r2d3','tidyverse','survMisc','jsonlite','DBI','ggplot2','googleVis','htmlwidgets','lubridate','magrittr','plotly','Rcpp','reshape','reshape2','rgl','stringr','tibble','tidyr','vcd','viridis','XML','xts'), quiet = TRUE)
renv::install('snowflakedb/dplyr-snowflakedb')
install.packages('datarobot')

renv::snapshot()
