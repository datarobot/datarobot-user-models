# Julia

This readme covers running Julia models with DRUM.  

Support was added for 
* Julia 1.5.4 (not testing with anything else)

Requirements
* python install
* python scikit learn
* python pandas
* python julia library 0.5.6

Be advised, start up time will be on the slower side since Julia follows a JIT compilation method.  See creating a system image below in performance testing to speed up start up time.  

Julia support in DRUM includes most mature interfaces available in Julia MLJ, specifically 
* [Clusting.jl](https://github.com/JuliaStats/Clustering.jl)
* [DecisionTree.jl](https://github.com/bensadeghi/DecisionTree.jl)
* [GLM.jl](https://github.com/JuliaStats/GLM.jl)
* [LIBSVM.jl](https://github.com/mpastell/LIBSVM.jl)
* [MLJModels.jl](https://github.com/alan-turing-institute/MLJModels.jl)
* [MultivariateStats.jl](https://github.com/JuliaStats/MultivariateStats.jl)
* [XGBoost.jl](https://github.com/dmlc/XGBoost.jl)
* [ScikitLearn.jl](https://github.com/cstjean/ScikitLearn.jl)

The expectation for use of DRUM with you Julia model artifacts
* you julia model leverages an MLJ interface
* your model has been serialized with a `.jlso` extension.
* your model may be instantiated with the `MLJ.machine("path_to_your_model.jlso")`

If the above are not true, you will need to use `custom.jl` file to create hooks that DRUM will leverage when using your model. 

Hooks currently supported
* `init`
* `read_input_data`
* `load_model`
* `transform`
* `score`
* `unstructured_predict`
* `post_process`

# Setup

This has only been confirmed to run with Julia 1.5.4 

Install Julia 1.5.4 and make sure it is available on the `PATH` environment variable

`pip install julia==0.5.6`

from python run 
```:python
import julia
julia.install()
```

confirm that it worked
```
import julia
jl = julia.Julia()
from julia import Base
print(Base.julia_cmd())
```

If you run into any issues here, it is likely due to different versions of python being use to build Julia PyCall vs what it is currently being called with (often seen with pyenv).  Another common issue is due to Python executables being statically linked to libpython.  See [pyjulia docs](https://pyjulia.readthedocs.io/en/latest/troubleshooting.html) for details.   

Also, I ran into multiple issues with Julia 1.6 and keep receiving segmentation faults.  

DRUM will look for the following environment variables

* `JULIA_PROJECT` - path to a julia project which would contain a `Manifest.toml` and a `Project.toml`.  If this is not set, a `Manifest.toml` and `Project.toml` bundled with DRUM will be used, and a Julia Project will be created based on these files and the necessary libraries will be installed.  If you use your own project you must install, at a minimum, the following libraries: `CSV`, `Pandas`, `PyCall`, and all the `MLJ` related libraries

* `JULIA_SYS_IMAGE` - path to a Julia system image.  Since Julia follows JIT compilation, it can feel very slow, and using a built system image with all your dependencies can speed things up considerable.

Neither variables are required and may constitute advanced usage of DRUM + Julia.  

```:julia
Pkg.activate("path-to-your-julia-project")
Pkg.add([
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
```

# Creating a system image 

To complete performance testing and validation of your Julia model with DRUM it is strongly encouraged to compile a system image of your julia environment.  If you do not, you run a strong chance of timeout with performance testing and validation.  This is due to the way the Julia follows a JIT compilation method.  

In Julia 
```
using Pkg, PackageCompiler
Pkg.activate("path-to-your-julia-project")
create_sysimage(
    [
        :MLJ,
        :MLJScikitLearnInterface,
        :MLJDecisionTreeInterface,
        :MLJClusteringInterface,
        :MLJGLMInterface,
        :MLJLIBSVMInterface,
        :MLJMultivariateStatsInterface,
        :MLJScikitLearnInterface,
        :MLJXGBoostInterface,
        :Pandas,
        :DataFrames,
        :CSV,
        :PyCall,
    ];
    sysimage_path = "path-to-write-sysimage/sys.dylib",
)
```

if on linux, `sys.so` instead of `sys.dylib` in `sysimage_path`.  

Then set environment variable `JULIA_SYS_IMAGE` and `JULIA_PROJECT`.


```
export JULIA_SYS_IMAGE=path-to-write-sysimage/sys.dylib
export JULIA_PROJECT=path-to-your-julia-project
```

# Examples

* Binary - The binary example is based on the iris dataset with target `Species`
* Multiclass - multiclass example based on the galaxy dataset with target `class`
* regression - grade dataset with target `Grade 2014`. 
* [Unstructured](./jl_unstructured/README.md)

### To run locally using 'drum'

To run these examples locally with `drum` installed, you must already have java 11 installed, or you can execute the examples with Docker.  

Paths are relative to `./datarobot-user-models/model_templates` unless fully qualified

### Binary 

`drum score --code-dir ./julia/jl_iris_binary --target-type binary --input ../../tests/testdata/iris_binary_training.csv --positive-class-label Iris-setosa --negative-class-label Iris-versicolor` 

### Multiclass 

`drum score --code-dir ./julia/jl_galaxy --target-type multiclass --class-labels GALAXY QSO STAR --input ../../tests/testdata/skyserver_sql2_27_2018_6_51_39_pm.csv` 

### Regression 

`drum score --code-dir ./julia/jl_grade --target-type regression --input ../../tests/testdata/juniors_3_year_stats_regression.csv`

#### Validation

```
drum validation --code-dir model_templates/julia/jl_iris_binary --target-type binary --input tests/testdata/iris_binary_training.csv --positive-class-label Iris-setosa --negative-class-label Iris-versicolor --verbose
```

#### Performance Testing 

```
drum perf-test --code-dir model_templates/julia/jl_iris_binary --target-type binary --input tests/testdata/iris_binary_training.csv --positive-class-label Iris-setosa --negative-class-label Iris-versicolor --verbose
```

#### Docker

You can either provide the path to the Dockerfile

Check out [Julia environment](public_dropin_environments/julia) in public_dropin_environments

`drum score --code-dir ./julia/jl_iris_binary --target-type binary --input ../../tests/testdata/iris_binary_training.csv --positive-class-label Iris-setosa --negative-class-label Iris-versicolor --docker ../../public_dropin_environments/julia_mlj/`

or provide the name of the docker image that has already been built [Julia MLJ Drop-In Environment](../../public_dropin_environments/julia_mlj/), for example, docker image is `drum_julia`.

`drum score --code-dir ./julia/jl_iris_binary --target-type binary --input ../../tests/testdata/iris_binary_training.csv --positive-class-label Iris-setosa --negative-class-label Iris-versicolor --docker drum_julia`

## DRUM Server

```
drum server --code-dir model_templates/julia/jl_iris_binary --target-type binary --positive-class-label Iris-setosa --negative-class-label Iris-versicolor --verbose --logging-level info --address 0.0.0.0:6789
```