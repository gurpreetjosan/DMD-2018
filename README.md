# DMD-2018

Theses are my best performing code for DMD shared task 2018. 

# DMD - Data Set

The data set was provided by organizers of the shared task and can be requested by sending your complete name, supervisor name, university name and an official email from your university e-mail address to vinayakumarr77@gmail.com. Due to privacy reasons, data set is not provided here.

# Running the Program on Your Own Data

Pre trained models for testing both malicious domain name and classifying domain name to respective botnet class. For testing malicious domain name call following script on command prompt
(Note: Replace line 45 i.e. foldername = 'test/Task 1/testing/Testing 1/test1.txt'  # replace with your test file path)

python3 test-models.py

For testing class of domain call following script on command prompt.

python3 test-multiclass-model.py

The output file will be produced in the same folder.

# Training Your own data

To train your own data run the program 
python3 malicious-dmd-2L.py 
and 
python3 multiclass-dmd-2l.py 

Note: Remember to change file path of training file according to your file structure.

