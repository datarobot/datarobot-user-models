
.PHONEY: verify run

create:
	mvn clean package

verify:
	python -m json.tool < component.json > /dev/null


run:
	run.sh
