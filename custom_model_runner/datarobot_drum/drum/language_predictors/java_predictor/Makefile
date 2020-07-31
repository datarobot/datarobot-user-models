
.PHONY: verify

build: clean verify
	mvn scala:compile -DdisplayCmd=true -DrecompileModel=all
	mvn clean package
	mvn clean

clean:
	rm -rf *.jar
