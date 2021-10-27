# MusiteDeep: a Deep-learning Framework for General and Kinase-specific Phosphorylation Site Prediction 

It is a Keras2.0 and Tensorflow version of MusiteDeep. It shares the same testdata folder which is under the MusiteDeep folder, please downloaded it and copy it to the MusiteDeep_keras2.0 foler to test the following experiments. 
# Installation

  - Installation has been tested in Linux and Mac OS X with Python 2.7. 
  - Since the package is written in python 2.7, [python 2.7](https://www.python.org/downloads/ ) with the pip tool must be installed first. 
MusiteDeep uses the following dependencies:
numpy,  scipy, pandas, h5py, keras version==2.1.2, tensorflow==1.3.0
You can install these packages first, by the following commands:

```sh
pip install pandas
pip install numpy
pip install scipy
pip install h5py
pip install -v keras==2.1.2
pip install tensorflow (or GPU supported tensorflow, refer to https://www.tensorflow.org/install/ for instructions)
```
 - This is the Tensorflow version, you must change the backend to TensorFlow.
If you have run Keras at least once, you will find the Keras configuration file at:
$HOME/.keras/keras.json
If it isn’t there, you can create it. 
Change the default configuration file into:
```sh
{	
    "image_dim_ordering": "th",
    "epsilon": 1e-07,
    "floatx": "float32",
    "backend": "tensorflow"
}
```
# Running on GPU or CPU

>If you want to use GPU, you also need to install [CUDA]( https://developer.nvidia.com/cuda-toolkit) and [cuDNN](https://developer.nvidia.com/cudnn); refer to their websites for instructions. 
CPU is only suitable for prediction not training. 

# For general users who want to perform human phosphorylation site prediction by our provided model :
cd to the MusiteDeep_Keras2.0/MusiteDeep folder which contains predict.py, train_general.py and train_kinase.py.  

#### For general phosphorylation site prediction using our pre-trained model, run:
```sh
python predict.py -input [custom predicting data in fasta format] -predict-type general -output [custom specified file for predicting results] 
```
##### Example:
```sh
python predict.py -input ../testdata/testing_proteins_STY.fasta -predict-type general -output result_test_general.txt -residue-types S,T,Y
```
You can change the type of sites for prediction by setting parameter ‘-residue-types’. For our general phosphorylation site prediction, only S, T and Y are acceptable. It takes about 15 minutes for running on CPU. The warnings can be ignored. The current model was trained by data "training_proteins_nonredundant_STY.fasta" which is under the ../testdata folder.

For details of other parameters, run:
```sh
python predict.py --help
```
or
```sh
python predict.py -h
```
#### For kinase-specific phosphorylation site prediction using our pre-trained model, run:
```sh
python predict.py -input [custom predicting data in fasta format] -predict-type kinase -kinase [custom specified kinase to predict] -output [custom specified file for predicting results]
```
##### Example:
Prediction for PKA:
```sh
python predict_batch.py -input ../testdata/testing_proteins_PKA.fasta -predict-type kinase -kinase PKA -output result_test_PKA.txt
```
Prediction for CDK:
```s
python predict_batch.py -input ../testdata/testing_proteins_CDK.fasta -predict-type kinase -kinase CDK -output result_test_CDK.txt
```
…
It takes about 5 minutes for running on CPU. The warnings can be ignored.
For details of other parameters, run:
```sh
python predict_batch.py --help
```
or
```sh
python predict_batch.py -h
```
The current models were trained by data "trainning_proteins_XXX.fasta", which is under the testdata folder. Here XXX indicates the kinase name.
# For advanced users who want to perform training and prediction by using their own data:

#### For custom training:
For custom general training using user’s training data:
```sh
python train_general.py -input [custom training data in fasta format] -output-prefix [prefix of pre-trained model] -residue-types [custom specified residue types]
```
For details of other parameters, run:
```sh
python train_general.py --help
```
or
```sh
python train_general.py -h
```
Examples will be shown together with other commands below.

For custom kinase-specific training (or performing transfer learning in which train models from several pre-trained models) from users’ training data:
To do so, the background models from one custom general data must be trained first by train_general.py, then train the kinase-specific models by using the background models to initialize the weights in the kinase-specific models. 
```sh
python train_kinase.py -input [custom training data in fasta format] -background-prefix [prefix of pre-trained model] -output-prefix [prefix of output files]
```
For details of other parameters, run:
```sh
python train_kinase.py --help
```
or 
```sh
python train_kinase.py -h
```
Examples will be shown together with other commands below.

#### Custom prediction from custom general models and custom kinase-specific models:
```sh
python predict_batch.py -input [custom predicting data in fasta format] -predict-type custom -model-prefix [prefix of pre-trained model] -output [custom specified file for predicting results] 
```
##### Examples for custom training and prediction from custom models.

When you have a lot of protein sequences in fasta format and you have changed the fasta format by adding “#” to the sites which are annotated sites of a specific PTM (use # to indicate positive sites), you can do custom general training. Taking the training of a general phosphorylation model as an example ,  training_proteins_nonredundant_STY.fasta in the ‘testdata’ folder is your training data, you can train a general model with prefix ‘custom_general’ and focusing on residues S,T, you can run the following command: 
```sh
python train_general.py -input ../testdata/training_proteins_nonredundant_STY.fasta -output-prefix custom_general_ST -residue-types S,T -nclass=5
```
In this command, since the parameter nclass is specified as 1 (to save time) only one general classifier will be trained. The default value for nclass is 5. And since the ‘-residue-types’ is set as S,T, only fragments center on S and T will be considered and used to train the model. Note that all the residues specified by this parameter will be trained in one model. So that S/T and Y cannot be used to train one model. For the model focusing on Y, a separate model need to be trained.
When you only have a small sample of protein sequence data, it is better to train general phosphorylation models first before training a kinase-specific phosphorylation model, then use the general phosphorylation models to initialize weights for the kinase-specific model, which is the concept of transfer learning.  Taking the training of PKA-specific phosphorylation model as an example, run the following command:
```sh
python train_kinase.py -input ../testdata/training_proteins_PKA.fasta  -background-prefix custom_general_ST -output-prefix custom_PKA -nclass=5
```
Here, custom_general_ST is the prefix of the pre-trained model by using the general phosphorylation training data ‘training_proteins_nonredundant_STY.fasta’ in the former command. You can also specify the number of the last layers to be randomly initialized by setting the parameter ‘-transferlayer’. The default value of ‘transferlayer’ is 1. If you don’t specify the residue type by parameter ‘-residue-types’, the same residues will be focused as in the general model.  You can set a different ‘-redisue-types’ from the general model. This is for training the general phosphorylation model of residue Y by using the general phosphorylation model for residues S and T as the background model to initialize weights in the new model. In this way, the performance of general phosphorylation models of residue Y is improved. Here is an example of training phosphorylation model for residue Y:
```sh
python train_kinase.py -input ../testdata/training_proteins_nonredundant_STY.fasta -background-prefix custom_general_ST -output-prefix custom_general_Y -nclass=5 -residue-types Y -transferlayer 0
```
This time only fragment’s center on Y will be considered and used to train the model.

##### Example of prediction from a custom general PTM model for residues S/T and Y:

```sh
python predict.py -input ../testdata/testing_proteins_STY.fasta -predict-type custom -model-prefix custom_general_ST -output custom_general_results.txt -residue-types S,T
python predict.py -input ../testdata/testing_proteins_STY.fasta -predict-type custom -model-prefix custom_general_Y -output custom_general_results.txt -residue-types Y
```
##### Example of prediction from a custom kinase-specific PTM model:

```sh
python predict.py -input ../testdata/testing_proteins_PKA.fasta -predict-type custom -model-prefix custom_PKA -output custom_PKA_results.txt 
```
### Training and testing data used for paper (Fig.4) is provided in the folder of testdata.
testing_proteins_ST.fasta is the testing data for S and T (annotated after 2008).

trainning_proteins_nonredundant_50_ST.fasta is the training data for S and T with no more than 50% identity with the testing data. trainning_proteins_nonredundant_10_ST.fasta is the training data for S and T with no more than 10% identity with the testing data. S or T followed by "#" indicates the positive sites.

### Evaluation data for Kinase-specific models is provided in the folder of testdata.
We used 5-fold cross-validation to evaluate the performance of MusiteDeep in the paper. The un-divided data set can be found in the folder of testdata by combining "trainning_proteins_XXX" with "testing_proteins_XXX_withannotation", here XXX indicates the name of a specific kinase family.


### Citation：
Please cite the following paper for using MusiteDeep:
Duolin Wang, Shuai Zeng, Chunhui Xu, Wangren Qiu, Yanchun Liang, Trupti Joshi, Dong Xu*, MusiteDeep: a Deep-learning Framework for General and Kinase-specific Phosphorylation Site Prediction.

License
----
GNU v2
