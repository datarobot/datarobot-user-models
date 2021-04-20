module Custom

using MLJ, MLJScikitLearnInterface, DataFrames

export MLJ, MLJScikitLearnInterface, transform

function transform(data, model)
    select!(data, Not(:class))
end

end
