#!/usr/bin/env Rscript

# Required for reticulate to work properly
Sys.setenv(MKL_THREADING_LAYER = "GNU")

install.packages('reticulate', repos='http://cran.us.r-project.org')

library("reticulate")

mlpiper <- import("mlpiper.components.external_component")
ext_comp <- mlpiper$ext_comp
params = ext_comp$get_params()
print(paste0("num_iter:    ", params["num_iter"]))
print(paste0("input_model: ", params["input_model"]))

parents_objs_list = ext_comp$get_parents_objs()
print(paste0("parents_objs_list lenght:", length(parents_objs_list)))
df1 = parents_objs_list[1]
str1 = parents_objs_list[2]

print(df1)
print(paste0("str1: ", str1))

# Generating a dummy dataset to demo L2 statistics
test = data.frame(temperature_1 = rnorm(100),
                  temperature_2 = rnorm(100),
                  pressure_1 = sample.int(10, 100, replace = TRUE),
                  pressure_2 = rep(c("A"    , "B", "C", "D"), 5))

# Setting the output of this component to be a single string object
ext_comp$set_output_objs("s1", "s2")
