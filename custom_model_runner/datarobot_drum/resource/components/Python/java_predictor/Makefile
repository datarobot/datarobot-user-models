
.PHONY: verify

build: clean verify
    mvn scala:compile -DdisplayCmd=true -DrecompileModel=all
	mvn clean package
	mvn clean

verify:
	python3 -m json.tool < component.json > /dev/null

clean:
	rm -rf *.jar
