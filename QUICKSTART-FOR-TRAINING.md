# :bowtie: Welcome to Custom Training Models :bowtie: 

This is a quick start guide.

You should be able to complete this guide without deviating at all. If you cannot, let us know!

## Background
Custom Training Models is a new feature where you, a programmer, can write some code to build a 
model, and then have that model get built within DataRobot, show up on the leaderboard, and then
be available for insights and deployment. Before now, the DataRobot Machine Learning Engineers
were the only ones who could write new models in DataRobot, but that all changes, starting now. 

## Steps

### Get your environment set up
1. The first thing we need to do is to get a python environment working. This guide shows you how
to do this one way, using pyenv. One way to install pyenv is to run `curl https://pyenv.run | bash`.
 Check out the github repo for other manners `https://github.com/pyenv/pyenv`
2. Put the lines bellow on your ~/.bashrc file:

```
export WORKON_HOME=~/.ve
export PROJECT_HOME=~/workspace
eval "$(pyenv init -)"
```

3. Install a version of python `pyenv install 3.6.0`
4. Create a virtual environment `pyenv virtualenv 3.6.0 custom-training-models`
5. Now that you're in a brand new virtual env, we're going to go install the dependencies we need
to test locally
```
pip install -r public_dropin_environments/python3_sklearn/dr_requirements.txt \
 -r public_dropin_environments/python3_sklearn/dr_requirements.txt
```
6. Theres no step 6! Thats it. 
### Test that your code works locally
7. To do this, we're going to be using the tester tool developed by DataRobot engineers to make 
sure your model is in tip top shape to be uploaded into the app. This tool is called DRUM. Here's
how to run it
```
drum fit --code-dir model_templates/python3_sklearn_training \
--input tests/testdata/iris_binary_training.csv --target Species --verbose
```
This is going to build a model with the code in the `python3_sklearn_training`, and then use
that model to make predictions using the training data it fit with. 
### Upload your code into DataRobot
8. Now that we've made sure the model can run, it's time to add it into DataRobot. To do this, 
go to `staging.datarobot.com`. 
9. First, make a project with the data file we just tested the model with. Start the project in 
manual mode.
10. Make sure that the feature flag `Enable Custom Training Models` is checked off. 
11. You should see a top level tab called the `Model Registry` click on that, and then go into the
custom model workshop
12. Create a new custom training model with the target type Binary. 
13. After this, drag and drop the contents of the code directory into the custom model file drop
14. Click on the big `Make Custom Blueprint` button
## Train your model
15. Now we can head over to the `Repository` tab, into the `Custom Blueprints` subtab
16. You should see the blueprint you just created. You can check the box next to it and then 
train it on your project. 

Thats it! This process has some rough edges still, but it will get easier and easier over time. 
Once your model is done training, you can check out how the model performs on the data provided 
by looking through the insights, or you can deploy the model as you would a normal DataRobot 
model. 
