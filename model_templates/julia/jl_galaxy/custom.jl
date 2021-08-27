module Custom

using MLJ, MLJScikitLearnInterface, DataFrames

export MLJ, MLJScikitLearnInterface, transform

function transform(data, model)
    if "Species" in names(data)
        data = select!(data, Not(:Species))
    elseif "MEDV" in names(data)
        data = select!(data, Not(:MEDV))
    elseif "class" in names(data)
        data = select!(data, Not(:class))
    end
    data = coalesce.(data, 0)
    return data
end

end
