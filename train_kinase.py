#python train_kinase.py -input train.fasta -background-prefix -output-prefix output 
#############background model must be trained first
import sys
import os
import pandas as pd
import numpy as np
import argparse

def main():
    parser=argparse.ArgumentParser(description='MusiteDeep custom training tool for kinase-specific PTM prediction.')
    parser.add_argument('-input',  dest='inputfile', type=str, help='training data in fasta format. Sites followed by "#" is a positive site for a specific PTM prediction.', required=True)
    parser.add_argument('-background-prefix',  dest='backgroundprefix', type=str, help='prefix of the pre-trained model by using general PTM data. It indicates two files [-background-prefix]_HDF5model and [-background-prefix]_parameters. If donnot have these files, please run train_general.py to get one from general PTM data, or you can just run train_general.py without background model using the kinase-specific data.', required=True)
    parser.add_argument('-output-prefix',  dest='outputprefix', type=str, help='prefix of output files (model and parameters of model) for prediction.', required=True)
    parser.add_argument('-valinput',  dest='valfile', type=str, help='validation data in fasta format if any. It will randomly select 10 percent of samples from the training data as a validation data set, if no validation file is provided.', required=False,default=None)
    parser.add_argument('-nclass',  dest='nclass', type=int, help='number of classifiers to be trained for one time. [Default:5]', required=False, default=5)
    parser.add_argument('-maxneg',  dest='maxneg', type=int, help='maximum iterations for each classifier which controls the maximum copy number of the negative data which has the same size with the positive data. [Default: 30]', required=False, default=30)
    parser.add_argument('-nb_epoch',  dest='nb_epoch', type=int, help='number of epoches for one bootstrap step. It is invalidate, if earlystop is set.', required=False, default=None)
    parser.add_argument('-earlystop',  dest='earlystop', type=int, help='after the \'earlystop\' number of epochs with no improvement the training will be stopped for one bootstrap step. [Default: 20]', required=False, default=20)
    parser.add_argument('-inputweights',  dest='inputweights', type=int, help='Initial weights saved in a HDF5 file.', required=False, default=None)
    parser.add_argument('-backupweights',  dest='backupweights', type=int, help='Set the intermediate weights for backup in a HDF5 file.', required=False, default=None)
    parser.add_argument('-transferlayer',  dest='transferlayer', type=int, help='Set the last \'transferlayer\' number of layers to be randomly initialized.', required=False, default=1)
    parser.add_argument('-residue-types',  dest='residues', type=str, help='Residue types that this model should focus on. For multiple residues, seperate each with \',\'. Note: all the residues specified by this parameter will be trained in one model.', required=False, default=None)
    
    
    args = parser.parse_args()
    inputfile=args.inputfile;
    backgroundprefix=args.backgroundprefix;
    backgroundmodel=backgroundprefix+str("_HDF5model")
    backgroundparameter=backgroundprefix+str("_parameters")
    transferlayer=args.transferlayer
    try:
        f= open(backgroundparameter, 'r')
    except IOError:
        print("cannot open "+ backgroundparameter+" ! Please run train_general.py to get one from general PTM data, or you can just run train_general.py without the background models using kinase-specific data.\n")
    else:
         parameters=f.read()
         f.close()
    
    outputprefix=args.outputprefix;
    valfile=args.valfile;
    nclass=args.nclass;
    nclass=int(nclass)
    nclass_init=parameters.split("\t")[0]
    nclass_init=int(nclass_init)
    window=parameters.split("\t")[1]
    window=int(window)
    residues_specify=args.residues;
    if(residues_specify is None):
       residues=parameters.split("\t")[2]
    else:
       residues=residues_specify
    maxneg=args.maxneg;
    np_epoch2=args.nb_epoch;
    earlystop=args.earlystop;
    inputweights=args.inputweights;
    backupweights=args.backupweights;
    outputmodel=outputprefix+str("_HDF5model");
    outputparameter=outputprefix+str("_parameters");
    
    try: 
       output = open (outputparameter,'w')
    except IOError:
       print('cannot write to ' + outputparameter+ "!\n")
       exit()
    else:
        output.write("%d\t%d\t%s\tkinase-specific\t%d\n" % (nclass,window,residues,nclass_init))
    
    from methods.Bootstrapping_allneg_continue_val import bootStrapping_allneg_continue_val
    from methods.EXtractfragment_sort import extractFragforTraining
    residues=residues.split(",")
    trainfrag=extractFragforTraining(inputfile,window,'-',focus=residues)
    if(valfile is not None):
        valfrag=extractFragforTraining(valfile,window,'-',focus= residues)
    else:
        valfrag=None;
    
    for ini in range(nclass_init):
        background=backgroundmodel+'_class'+str(ini)
        for bt in range(nclass):
            models=bootStrapping_allneg_continue_val(trainfrag.as_matrix(),valfile=valfrag,
                                                     srate=1,nb_epoch1=1,nb_epoch2=np_epoch2,maxneg=maxneg,
                                                     outputweights=backupweights,
                                                     earlystop=earlystop,
                                                     forkinas=True,
                                                     inputweights=background,
                                                     transferlayer=transferlayer)
            models.save_weights(outputmodel+'_ini'+str(ini)+'_class'+str(bt),overwrite=True)
    
    
if __name__ == "__main__":
    main()         
   