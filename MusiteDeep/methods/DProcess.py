import string
import re
import pandas as pd
import numpy as np
import keras.utils.np_utils as kutils

def convertSampleToProbMatr(sampleSeq3DArr): #changed add one column for '1'
    """
    Convertd the raw data to probability matrix
    
    PARAMETER
    ---------
    sampleSeq3DArr: 3D numpy array
       X denoted the unknow amino acid.
    
    
    probMatr: Probability Matrix for Samples. Shape (nb_samples, 1, nb_length_of_sequence, nb_AA)
    """
    
    letterDict = {}
    letterDict["A"] = 0
    letterDict["C"] = 1
    letterDict["D"] = 2
    letterDict["E"] = 3
    letterDict["F"] = 4
    letterDict["G"] = 5
    letterDict["H"] = 6
    letterDict["I"] = 7
    letterDict["K"] = 8
    letterDict["L"] = 9
    letterDict["M"] = 10
    letterDict["N"] = 11
    letterDict["P"] = 12
    letterDict["Q"] = 13
    letterDict["R"] = 14
    letterDict["S"] = 15
    letterDict["T"] = 16
    letterDict["V"] = 17
    letterDict["W"] = 18
    letterDict["Y"] = 19
    letterDict["-"] =20 ##add -
    AACategoryLen = 21 ##add -
    
    probMatr = np.zeros((len(sampleSeq3DArr), 1, len(sampleSeq3DArr[0]), AACategoryLen))
    
    
    sampleNo = 0
    for sequence in sampleSeq3DArr:
    
        AANo	 = 0
        for AA in sequence:
            
            if not AA in letterDict:
                probMatr[sampleNo][0][AANo] = np.full((1,AACategoryLen), 1.0/AACategoryLen)
            
            else:
                index = letterDict[AA]
                probMatr[sampleNo][0][AANo][index] = 1
                
            AANo += 1
        sampleNo += 1
    
    return probMatr
    


def convertSampleToIndex(sampleSeq3DArr):
	"""
	Convertd the raw data to probability matrix
	
	PARAMETER
	---------
	sampleSeq3DArr: 3D numpy array
		X denoted the unknow amino acid.
	
	
	probMatr: Probability Matrix for Samples. Shape (nb_samples, 1, nb_length_of_sequence, nb_AA)
	"""
	
	letterDict = {}
	letterDict["A"] = 1
	letterDict["C"] = 2
	letterDict["D"] = 3
	letterDict["E"] = 4
	letterDict["F"] = 5
	letterDict["G"] = 6
	letterDict["H"] = 7
	letterDict["I"] = 8
	letterDict["K"] = 9
	letterDict["L"] = 10
	letterDict["M"] = 11
	letterDict["N"] = 12
	letterDict["P"] = 13
	letterDict["Q"] = 14
	letterDict["R"] = 15
	letterDict["S"] = 16
	letterDict["T"] = 17
	letterDict["V"] = 18
	letterDict["W"] = 19
	letterDict["Y"] = 20
	letterDict["-"] = 21
	letterDict["X"] = 0
	probMatr = np.zeros((len(sampleSeq3DArr),len(sampleSeq3DArr[0])))
	
	sampleNo = 0
	for sequence in sampleSeq3DArr:
		AANo	 = 0
		for AA in sequence:
			probMatr[sampleNo][AANo]= letterDict[AA]
			AANo += 1
		sampleNo += 1
	
	return probMatr
	


def convertSampleToVector2DList(sampleSeq3DArr, nb_windows, refMatrFileName):
	"""
	Convertd the raw data to probability matrix
	PARAMETER
	---------
	sampleSeq3DArr: 3D List
		List -  numpy matrix(3D)
	Sample List: List (nb_windows, nb_samples, SEQLen/nb_windows , 100)
	"""
	
	rawDataFrame = pd.read_table(refMatrFileName, sep='\t', header=None)
	
	raw_data_seq_index_df = pd.DataFrame({'seq' : rawDataFrame[0] , 'indexing':rawDataFrame.index})
	raw_data_seq_df_index_dict = raw_data_seq_index_df.set_index('seq')['indexing'].to_dict()

	
	nb_raw_data_frame_column = len(rawDataFrame.columns)
	
	nb_sample = sampleSeq3DArr.shape[0]
	len_seq = len(sampleSeq3DArr[1]) 
	re_statement =  ".{%d}" % (nb_windows)
	
	
	probMatr_list = []
	for tmp_idx in range(nb_windows):
		probMatr_list.append( np.zeros((nb_sample, int((len_seq - tmp_idx)/nb_windows) , 100)) )

	
	for sample_index, sample_sequence in enumerate(sampleSeq3DArr):
		
		if sample_index%10000 == 0:
			print( "%d / %d " % (sample_index, nb_sample))
		
		#start_time = time.time()
		seq = "".join(sample_sequence)
		
		for begin_idx in range(nb_windows):
			
			# Get sub-sequence
			sub_seq_list = re.findall(re_statement, seq[begin_idx:])
			
			sub_seq_indexing_list = []
			for sub_seq in sub_seq_list:
				if sub_seq in raw_data_seq_df_index_dict:
					sub_seq_indexing_list.append( raw_data_seq_df_index_dict[sub_seq] )
				else:
					sub_seq_indexing_list.append( raw_data_seq_df_index_dict['<unk>'] )

			matrix_arr = rawDataFrame.loc[ sub_seq_indexing_list ][ range(1, nb_raw_data_frame_column)].as_matrix()
			for idx_i in range(matrix_arr.shape[0]):
				for idx_j in range(matrix_arr.shape[1]):
					probMatr_list[begin_idx][sample_index][idx_i][idx_j] = matrix_arr[idx_i][idx_j]

		#print("2. --- %s seconds ---" % (time.time() - start_time))
		

	return probMatr_list

def convertSampleToDoubleVec(sampleSeq3DArr, nb_neibor):
    letterDict = {}
    letterDict["A"] = 0
    letterDict["C"] = 1
    letterDict["D"] = 2
    letterDict["E"] = 3
    letterDict["F"] = 4
    letterDict["G"] = 5
    letterDict["H"] = 6
    letterDict["I"] = 7
    letterDict["K"] = 8
    letterDict["L"] = 9
    letterDict["M"] = 10
    letterDict["N"] = 11
    letterDict["P"] = 12
    letterDict["Q"] = 13
    letterDict["R"] = 14
    letterDict["S"] = 15
    letterDict["T"] = 16
    letterDict["V"] = 17
    letterDict["W"] = 18
    letterDict["Y"] = 19
    
    
    double_letter_dict = {}
    for key_row in letterDict:
        for key_col in letterDict:
            idx_row = letterDict[key_row]
            idx_col = letterDict[key_col]
            
            final_key = key_row    + key_col
            final_idx = idx_row*20 + idx_col
            
            double_letter_dict[final_key] = final_idx
    
    
    probMatr = np.zeros((len(sampleSeq3DArr), 1, len(sampleSeq3DArr[0])-nb_neibor, len(double_letter_dict)))

    
    sampleNo = 0
    for sequence in sampleSeq3DArr:
    
        nb_sub_AA   = 0
        sequence = sequence.tolist()
        for idx in range(len(sequence)-nb_neibor):
            
            sub_AA = ("").join( sequence[idx:idx+nb_neibor+1] )
            
            if sub_AA in double_letter_dict:
                index = double_letter_dict[sub_AA]
                probMatr[sampleNo][0][nb_sub_AA][index] = 1
            print(sub_AA)
            break
            nb_sub_AA += 1
        break
        sampleNo += 1

    
    return probMatr
    
    
    

def convertRawToXY(rawDataFrame, refMatrFileName="", nb_windows=3, codingMode=0):#rawDataFrame is numpy.ndarray
    """
    convertd the raw data to probability matrix and target array 
    
    
    #Output:
    probMatr: Probability Matrix for Samples. Shape (nb_samples, 1, nb_length_of_sequence, nb_AA)
    targetArr: Target. Shape (nb_samples)
    """
    
    
    #rawDataFrame = pd.read_table(fileName, sep='\t', header=None).values
    
    targetList = rawDataFrame[:, 0]
    targetList[np.where(targetList==2)] = 0
    targetArr = kutils.to_categorical(targetList)
    
    sampleSeq3DArr = rawDataFrame[:, 1:]
    
    if codingMode == 0:
        probMatr = convertSampleToProbMatr(sampleSeq3DArr)
    elif codingMode == 1:
        probMatr = convertSampleToVector2DList(sampleSeq3DArr, nb_windows, refMatrFileName)
    elif codingMode == 2:
        probMatr = convertSampleToDoubleVec(sampleSeq3DArr, 1)
    return probMatr, targetArr
     


def convertRawToIndex(rawDataFrame):
	#rawDataFrame = pd.read_table(fileName, sep='\t', header=None).values
	
	targetList = rawDataFrame[:, 0]
	targetList[np.where(targetList==2)] = 0
	targetArr = kutils.to_categorical(targetList)

	sampleSeq3DArr = rawDataFrame[:, 1:]
	
	index = convertSampleToIndex(sampleSeq3DArr)
	
	
	return index, targetArr
	


def convertRawToX(fileName, refMatrFileName="", nb_windows=3, codingMode=0):
	"""
	convertd the raw data to probability matrix
	
	
	#Output:
	probMatr: Probability Matrix for Samples. Shape (nb_samples, 1, nb_length_of_sequence, nb_AA)
	"""
	
	
	rawDataFrame = pd.read_table(fileName, sep='\t', header=None).values
	
	sampleSeq3DArr = rawDataFrame[:, 0:]
	
	if codingMode == 0:
		probMatr = convertSampleToProbMatr(sampleSeq3DArr)
	elif codingMode == 1:
		probMatr = DProcess.convertSampleToVector2DList(sampleSeq3DArr, nb_windows, refMatrFileName)
	
	
	return probMatr
