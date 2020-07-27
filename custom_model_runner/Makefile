
.PHONY: clean wheel

dist: clean java_components wheel

wheel: clean
	python3 setup.py bdist_wheel

java_components:
	cd datarobot_drum/drum/language_predictors/java_predictor && $(MAKE)

clean:
	\rm -rf dist build datarobot_drum.egg-info
	find . -type d -name __pycache__ | xargs rm -rf
	find . -name *.pyc -delete

