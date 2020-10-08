# R2D2: A testing inference model

r2d2 inference model provides a quick way to test your environment.
r2d2 is not really a model, it is a testing program which interpret 
the prediction requests as actions to execute inside the drum environment.

r2d2 expects an input dataframe to contain the 2 columns: "cmd" and "arg".
The "cmd" column is supposed to contain the name of the command to run.
The "arg" column is supposed to contain an argument to pass to the command.

For example, the following csv will cause r2d2 to consume 1500 megabytes 
of memory 
    
    cmd,arg
    memory, 1500

More info about supported commands can be found in the next
section.

**Important** info:

* r2d2 only takes the command from the first row in the input data.
Following rows are ignored.
* r2d will not fail if input data does not contain the "cmd" and "arg". 
An error message will be printed to r2d2 stdout, but r2d2 will not fail
the prediction.

For example the following csv file, will only consume 2000 of memory:

    cmd,arg
    memory,2000
    memory,3000


## What kind of actions are supported:
r2d2 supports the following actions:
* Consume memory
* Throw exceptions
* Sleep for a given amount of time, potentially causing timeout.

### Use memory
This action will cause r2d2 to consume memory. The way to trigger that
action is to send the "memory" as the command in the "cmd" column of the 
prediction request. And the total amount of memory to consume in megabytes in
the "arg" column.

For example, use the following csv file to cause r2d2 consume 1000mb of
memory:

    cmd,arg
    memory,1000
 
Things to note:
   
 * Sending an arg value <= 0 in the arg column, will result in releasing the memory allocating so far
 * Sending the same number again will not allocate additional memory. Memory allocation can
   only be increased. For example, sending 2000 and then again 2000, will result in a single
   memory allocation of 2000mb.  

### Exception throwing
This action will cause r2d2 to throw an exception. To do that, send the "cmd"
column to "exception". Currently the "arg" column is ignored

For example, use the following csv file to cause an exception in predictions.
(Note: r2d2 will ignore the arg value, 33 in our case).

    cmd,arg
    exception, 33

### Timeout
This action will cause r2d2 to sleep for a given amount of seconds. To do that
set the "cmd" column value to "timeout", and set the "arg" column value to 
the number of seconds you would like r2d2 to sleep.

For example, use the following csv file to cause a timeout of 10 seconds.

    cmd, arg
    timeout, 10


## How to run r2d2 send predictions in server mode
First run drum in server mode. For example

    drum server --code-dir ./r2d2/  --address localhost:8999 --verbose

In this example we would like to cause the drum environment to consume
2000 megabytes (or so) of memory. So prepare a file like the following:

    cmd,arg
    memory,2000
    
Name your csv file memory.csv. 

You can now trigger r2d2 to consume 2000mb of memory by doing:

    curl -X POST --form "X=@./memory.csv" localhost:8999/predict/
    
 In the above example, the data in the file _memory.csv_ will be sent
 to the endpoint "localhost:8999/predict". DRUM will create a dataframe
 form that file and will forward it to the r2d2 model. The r2d2 will
 parse the incoming data and will consume 2000mb of memory. 
 
 ## Using the r2d2/custom.py to send commands
 
 r2d2/custom.py contains a main() function that if called can be used to send commands
 to a running r2d2 model. For example
 
     python r2d2/custom.py 