
.PHONY: verify

build: clean verify
	mvn clean install
	mvn clean

sonatype-deploy:
	@echo "Deploying DRUM Java Predictor API (Sonatype repository)"
	mvn clean deploy -f sonatype_pom.xml -P release-sonatype

clean:
	rm -rf *.jar
