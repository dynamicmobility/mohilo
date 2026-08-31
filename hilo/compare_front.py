# Read in a dataset file
# In that directory, make a csv that contains 
# trial | action | obj1 est | obj2 est | type
# 1     | [0, ..]| 0...     | 0...     | pareto (or antipareto)

# this csv shows the true pareto and antipareto set given the experiment dataset
# refer to scratch/compare_front.py to see how might work

# next, the script should start a new experiment dataset, called evaluation, which
# trials each action to the user, and uses the same objectives from hilo.hardware 
# or hilo.simulation. a CLI argument should be accepted called --reversed, which
# does the same thing except saved to an experiment dataset called evaluation_reversed

# this experiment code should look extremely similar to hilo/fit_mogp.py, except
# that there is now no GP. Please keep the code clean and as similar as possible 
# to this file

# all these files should be saved in the original experiment directory. log files
# should be named accordingly as well.

# let me know if you have questions.