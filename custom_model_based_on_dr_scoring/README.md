# Custom model implementing Predictor interface

1. cd ./custom_model_based_on_dr_scoring
2. mvn package   (currently it is set up for Java 11, I haven't try with Java 8)
3. install drum (pip instal datarobot-drum)
4. drum score --code-dir ./target/  --input ./data/for_cust_model.csv --target-type regression