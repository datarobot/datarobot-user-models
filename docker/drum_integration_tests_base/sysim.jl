using Pkg
@info "Activating project environment"
Pkg.activate(ENV["JULIA_PROJECT"])
@info "installing dependencies"
@time Pkg.add([
    "PackageCompiler",
    "MLJ",
    "MLJScikitLearnInterface",
    "MLJDecisionTreeInterface",
    "MLJClusteringInterface",
    "MLJGLMInterface",
    "MLJLIBSVMInterface",
    "MLJMultivariateStatsInterface",
    "MLJXGBoostInterface",
    "Pandas",
    "DataFrames",
    "CSV",
    "PyCall",
])

using PackageCompiler,
    MLJ,
    MLJScikitLearnInterface,
    MLJDecisionTreeInterface,
    MLJClusteringInterface,
    MLJGLMInterface,
    MLJLIBSVMInterface,
    MLJMultivariateStatsInterface,
    MLJScikitLearnInterface,
    MLJXGBoostInterface,
    Pandas,
    DataFrames,
    CSV,
    PyCall
@info "creating system image at $(ENV["JULIA_SYS_IMAGE"])"
@time create_sysimage(
    [
        :MLJ,
        :MLJScikitLearnInterface,
        :MLJDecisionTreeInterface,
        :MLJClusteringInterface,
        :MLJGLMInterface,
        :MLJLIBSVMInterface,
        :MLJMultivariateStatsInterface,
        :MLJXGBoostInterface,
        :Pandas,
        :DataFrames,
        :CSV,
        :PyCall,
    ];
    sysimage_path = ENV["JULIA_SYS_IMAGE"],
)
@info "Julia environment setup complete! \U0001F37E"
