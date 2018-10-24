Read Me
======================
Requirements
----------------------
* Anaconda 4.X (Python 3.5+)
* NumPy
* SciPy
* BLAS
* Pandas
* Theano
* Sklearn


Usage
----------------------
### Datasets
* Two sample datasets are provided with anonymization in data folder
* Two RecSys Challenge dataset can be downloaded at https://www.dropbox.com/sh/2jw72n03t6zyde9/AAD25moSH9qmiIHe1YGewUYea?dl=0
* All datasets in the same format can be used for static configuration (run_test.py)
* full 'course3' and 'news2' datasets ordered by time are used for adpative configuration (run_test_adapt.py); 


### Running experiments
1. Double check datasets in data folder, or download from the link above
2. static configuration:
   run_test.py: train model on trainig data and evaluates all testing views per 'session'.
   results and computation times will be output to both terminal and a file.
3. adaptive configuration:
   run_test_adapt.py evaluates all views per 'event' on full dataset (ordered by time)
   results will be output to both terminal and a file

