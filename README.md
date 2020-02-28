Summary
-------
This project directory processes results from CalSim3 and HEC5Q studies to
establish a non-linear relationship between flow and temperature operations
with the TensorFlow library.

Research
--------
A report detailing the theory of Artificial Neural Networks and its application
to the CalSim3 model is available on the U.S. Bureau of Reclamation website
via the following link:
https://www.usbr.gov/research/projects/researcher.cfm?id=2931

Requirements
------------
The procedure for training a temperature Artificial Neural Network (ANN)
requires Python and many popular libraries that come with [Anaconda for Python](https://www.anaconda.com/distribution/)
for Windows 10. Additionally, the [calsim_toolkit](https://github.com/jshannon-usbr/calsim_toolkit)
and [TensorFlow](https://www.tensorflow.org/install/pip) libraries to execute
the scripts in this project.

A CalSim3 baseline study and HEC5Q model developed to accepted CalSim3 inputs
are required to produce the training data for the temperature ANN. Contact the
individuals in the "Contact" section for valid models.

Procedure
---------
Change the working directory to "scripts" and execute the following scripts in
the described order:
1. calsim3_runs.py: Run variations of a CalSim3 baseline to establish flow
   regimes.
1a. (Optional) keswick_plot.py: View Shasta release flow variation relative to
     the baseline CalSim3 study.
2. hec5q_runs.py: Calculate temperature profiles based on CalSim3 flow regimes.
3. train_temperature_nn.py: Train the temperature ANN.

The resulting ANN model is produced in __trained_tnn/temperature_nn.py, with
the trained weights as input text files.

To connect the ANN to a CalSim3 model, a special version of WRIMS is required
to reference Python scripts. Contact Hao Xie at Hao.Xie@water.ca.gov for this
special version of WRIMS and details of how to connect the Python ANN model.

Contact
-------
Jim Shannon
U.S. Bureau of Reclamation
2800 Cottage Way
Sacramento, CA 95816
916-978-5078
jshannon@usbr.gov
