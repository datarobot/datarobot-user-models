using Base.Filesystem
# using MLJ
# using MLJScikitLearnInterface,
#     MLJDecisionTreeInterface,
#     MLJClusteringInterface,
#     MLJGLMInterface,
#     MLJLIBSVMInterface,
#     MLJMultivariateStatsInterface,
#     MLJXGBoostInterface
using Pandas
using DataFrames
using CSV

const REGRESSION_PRED_COLUMN_NAME = "Predictions"
const RUNNING_LANG_MSG = "Running environment language: Julia"
const CUSTOM_MODEL_FILE_EXTENSION = ".jlso"

global defined_hooks = Dict(
    "init" => false,
    "read_input_data" => false,
    "load_model" => false,
    "transform" => false,
    "score" => false,
    "unstructured_predict" => false,
    "post_process" => false,
)

## initialize

function init(code_dir, target_type)

    ## include custom.jl 
    custom_path = Filesystem.joinpath(code_dir, "custom.jl")
    if Filesystem.isfile(custom_path)
        custom_loaded = (
            try
                include(custom_path)
                @eval using Main.Custom
                @info "successfully included $custom_path"
                true
            catch e
                bt = backtrace()
                msg = sprint(showerror, e, bt)
                @error """\nload custom.jl failed"""
                @error msg 
                throw(Exception)
            end
        )
    else
        @info "no custom.jl detected"
        custom_loaded = false
    end


    if custom_loaded
        hooks = names(Custom)
        for hook in hooks
            hook_str = string(hook)
            if haskey(defined_hooks, hook_str)
                @info "$hook_str is defined"
                global defined_hooks[hook_str] = true
            end
        end
    end
    if defined_hooks["init"]
        ## no idea why this work :(
        @info "running init from defined hooks with code directory $code_dir"
        # @eval Custom.init($code_dir)
        Base.invokelatest(Custom.init, code_dir)
    end

end

function load_serialized_model(model_dir)

    model = nothing

    files = Filesystem.readdir(model_dir)
    artifacts = [x for x in files if endswith(lowercase(x), CUSTOM_MODEL_FILE_EXTENSION)]
    if length(artifacts) == 0
        @error """\nCould not find a serialized model artifact with $CUSTOM_MODEL_FILE_EXTENSION extension, supported by default Julia predictor.  
        If your artifact is not supported by default predictor, implement Custom.load_model hook."""
        throw(Exception)
    elseif length(artifacts) > 1
        @error """Multiple serialized model artifacts found: $(join(artifacts, ", "))"""
        throw(Exception)
    else
        artifact = artifacts[1]
        artifact_path = Filesystem.joinpath(model_dir, artifact)
        mlj_libs = Filesystem.joinpath(@__DIR__, "mlj_libs.jl")
        @info include(mlj_libs)
        ## thinking about loading necessary MLJ modules instead 
        ## of top leve.  should save some time and lighten lib 
        ## requirement on user if there is no intention to use MLJ
    end
    if defined_hooks["load_model"]
        @info "Loading model via hook"
        model = Custom.load_model(model_dir)
    else
        @info "Loading serialized model artifact found at $artifact_path"
        model = Main.MLJLibs.MLJ.machine(artifact_path)
    end

    return model
end

function predict_regression(data, model; kwargs...)
    if defined_hooks["score"]
        predictions = Custom.score(data, model; kwargs...)
    else
        predictions = Main.MLJLibs.MLJ.predict(model, data)
    end
    return Pandas.DataFrame(predictions, columns=[REGRESSION_PRED_COLUMN_NAME])
end

function predict_classification(
    data,
    model;
    positive_class_label=nothing,
    negative_class_label=nothing,
    class_labels=nothing,
    target_type=nothing,
)
    @debug "postive_class_label = $positive_class_label"
    @debug "negative_class_label = $negative_class_label"
    @debug "class_labels = $class_labels"
    if !isnothing(positive_class_label) & !isnothing(negative_class_label)
        labels = [positive_class_label, negative_class_label]
    elseif !isnothing(class_labels)
        labels = class_labels
    else
        throw("something is not right")
    end
    yhat = Main.MLJLibs.MLJ.predict(model, data)
    probs = Main.MLJLibs.MLJ.pdf(yhat, labels)
    return Pandas.DataFrame(probs, columns=labels)
end

predictors = Dict(
    "binary" => predict_classification,
    "multiclass" => predict_classification,
    "regression" => predict_regression,
    "anomaly" => predict_regression,
)

function model_predict(data, model; kwargs)
    target_type = kwargs.target_type
    if !haskey(predictors, target_type)
        throw("Target type $target_type is not supported by Julia predictor")
    else
        predictors[target_type](data, model; kwargs...)
    end
end

function outer_predict(
    target_type;
    binary_data=nothing,
    mimetype=nothing,
    model=nothing,
    positive_class_label=nothing,
    negative_class_label=nothing,
    class_labels=nothing,
)

    if defined_hooks["read_input_data"]
        data = Custom.read_input_data(binary_data)
    elseif !isnothing(mimetype) && mimetype == "text/mtx"
        @error "MTX is not yet supported"
        throw(Exception)
    else
        data = CSV.read(
            IOBuffer(binary_data),
            DataFrames.DataFrame,
            delim=","
        )
    end

    if isnothing(model)
        model = load_serialized_model()
    end

    if defined_hooks["transform"]
        data = Custom.transform(data, model)
    end

    kwargs = (
        positive_class_label = positive_class_label,
        negative_class_label = negative_class_label,
        class_labels = class_labels,
    )

    if defined_hooks["score"]
        predictions = Custom.score(data, model; kwargs)
    else
        kwargs = merge(kwargs, (target_type = target_type,))
        predictions = model_predict(data, model; kwargs)
    end

    if defined_hooks["post_process"]
        predictions = Custom.post_process(predictions, model)
    end

    return Pandas.DataFrame(predictions)
end

function predict_unstructured(data; model=nothing, mimetype=nothing, charset=nothing, query=nothing)
    kwargs = (mimetype = mimetype, charset = charset, query = query)
    predictions = Custom.score_unstructured(data; model, kwargs)
    return predictions
end