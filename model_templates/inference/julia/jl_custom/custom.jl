
module Custom

using Flux
using BSON:@load
using DataFrames
using Pandas
using Pkg

export Flux, load_model, transform, score
# export loadModel, score

function load_model(code_dir)  
    @load "$(code_dir)/mymodel.bson" model
    return model
end

function transform(df, model)
    try
        df = DataFrames.select(df, Not(:MEDV))
    catch
        nothing
    end
    return df
end

function score(df, model; kwargs...)
    X = convert(Matrix, df)  
    yhat = model(X')
    yhat = yhat'
    return Pandas.DataFrame(yhat, columns = ["Predictions"])
end

end