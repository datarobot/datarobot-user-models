module Custom

using MLJ, MLJScikitLearnInterface, DataFrames

export MLJ, MLJScikitLearnInterface, transform

function transform(data, model)
    select!(data, Not(:Species))
end


end
