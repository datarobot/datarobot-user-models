module Custom

using PyCall 
export load_model, unstructured_predict

function to_bytes(n::Integer; bigendian=true, len=sizeof(n))
    bytes = Array{UInt8}(undef, len)
    for byte in (bigendian ? (1:len) : reverse(1:len))
        bytes[byte] = n & 0xff
        n >>= 8
    end
    return bytes
end

function load_model(code_dir)  
    return nothing
end

function score_unstructured(data; model = nothing, kwargs = nothing)
    println("Model: $model")
    println("Incoming content type params: $kwargs")
    println("Incoming data type $(typeof(data))")
    println("Incoming data: $data")
    println("Incoming query params: $(kwargs.query)")
    
    if haskey(kwargs.query, "ret_mode")
        ret_mode = kwargs.query["ret_mode"]
    else 
        ret_mode = nothing
    end

    word_count = split( replace( data, "\n" => " "), " ")
    word_count = length(word_count)

    if ret_mode == "binary"  
        ret_data = to_bytes(word_count)
        ret_kwargs = PyCall.PyDict( Dict("mimetype" => "application/octet-stream"))
        ret = PyCall.pybytes(ret_data), ret_kwargs
    else 
        ret_kwargs = PyCall.PyDict( Dict("mimetype" => kwargs.mimetype)) 
        ret = string(word_count)
        #, ret_kwargs
    end
    println("outgoing")
    println(ret)
    return ret
end

end