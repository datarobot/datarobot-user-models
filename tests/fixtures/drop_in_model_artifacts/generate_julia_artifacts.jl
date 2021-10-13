using DataFrames
using MLJ
using CSV
using Logging

function make_artifacts(data_path, target, target_type, export_location, estimator)
    @info "loading $data_path"
    df = DataFrame(CSV.File(data_path))
    y, X = unpack(df, ==(Symbol(target)), colname -> true);
    @info "dropping id columns if present"
    try
        X = select!(X, Not(Symbol("objid")))
    catch
    end
    try
        X = select!(X, Not(Symbol("Id")))
    catch
    end
    X = coalesce.(X,-99999)
    tree = estimator() 
    if target_type == "regression"
        @info "training regression estimator"
        ml = machine(tree, X, float(y))
    else
        @info "training classification estimator"
        ml = machine(tree, X, categorical(y))
    end
    train, test = partition(eachindex(y), 0.7, shuffle=true); # 70:30 split
    fit!(ml, rows=train, verbosity=0)
    @info "writing model to disk at $export_location"
    MLJ.save(export_location, ml)
end

## regression model details
regression_data = "tests/testdata/juniors_3_year_stats_regression.csv"
regression_target = "Grade 2014"
regression_output = "tests/fixtures/drop_in_model_artifacts/grade_regression.jlso"

binary_data = "tests/testdata/iris_binary_training.csv"
binary_target = "Species"
binary_output = "tests/fixtures/drop_in_model_artifacts/iris_binary.jlso"

multiclass_data = "tests/testdata/skyserver_sql2_27_2018_6_51_39_pm.csv"
multiclass_target = "class"
multiclass_output = "tests/fixtures/drop_in_model_artifacts/galaxy.jlso"

RandomForestRegressor = @load RandomForestRegressor pkg="ScikitLearn"
RandomForestClassifier = @load RandomForestClassifier pkg="ScikitLearn"

make_artifacts(regression_data, regression_target, "regression", regression_output, RandomForestRegressor)
make_artifacts(binary_data, binary_target, "classification", binary_output, RandomForestClassifier)
make_artifacts(multiclass_data, multiclass_target, "classification", multiclass_output, RandomForestClassifier)