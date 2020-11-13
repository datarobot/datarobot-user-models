# Performance testing DRUM with Locust
## About
Here you can find some materials describing how to run Locust to test DRUM performance. 

## Content
### Suits
`Suit` == `locustfile` is the main entity that implements the task in Locust framework.
Currently there are two of them prepared to run tasks on DRUM:
- [predict suit](suits/predict.py) - implements structured predict request to DRUM;
- [unstructured predict suit](suits/predict_unstructured.py) - implements unstructured predict request to DRUM;

### Scripts
- [start-locust-docker.sh](scripts/start-locust-docker.sh) - the main script that configures and starts Locust using provided parameters.
#### How script works
It utilizes official Locust docker container to start Locust with the provided parameters.
#### Usage
`./start-locust-docker.sh -l "-u 5 -r 5 -H http://localhost:6788 -t 5 --headless --csv drum" -d ../testdata/boston_housing.csv`
##### Parameters
- -l - string composed of native Locust parameters, check `locust --help` or [https://docs.locust.io/en/stable/configuration.html](https://docs.locust.io/en/stable/configuration.html)
- -d - samples dataset to send
- -s - number of samples to send

> Note: currently structured predict locustfile is hardcoded to be used.

#### Running DRUM
This is user's responsiblity to run DRUM and provide proper url address to Locust either in param `-H http://localhost:6788` or UI.
It is recommended to run DRUM in a [public dropin environment](../../public_dropin_environments) docker container.
These environments are enabled to run DRUM with:
- single threaded Flask server
- single worker uwsgi + nginx server
- multi worker uwsgi + nginx server    
 

### Empty unstructured model
This is an implementation of empty unstructured model, which can be used when testing requires no data to be sent in request/response.