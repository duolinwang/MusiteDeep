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

def convertSampleToPhysicsVector(sampleSeq3DArr):
	"""
	Convertd the raw data to physico-chemical property
	
	PARAMETER
	---------
	sampleSeq3DArr: 3D numpy array
		X denoted the unknow amino acid.
	
	
	probMatr: Probability Matrix for Samples. Shape (nb_samples, 1, nb_length_of_sequence, nb_AA)
	"""
	
	letterDict = {} #hydrophobicty, hydrophilicity, side-chain mass, pK1 (alpha-COOH), pK2 (NH3), PI, Average volume of buried residue, Molecular weight, Side chain volume, Mean polarity
	letterDict["A"] = [0.62,	-0.5,	15,	2.35,	9.87,	6.11,	91.5,	89.09,	27.5,	-0.06]
	letterDict["C"] = [0.2900,	-1.0000,	47.0000,    1.7100,   10.7800,    5.0200,	117.7,	121.15,	44.6,	1.36]
	letterDict["D"] = [-0.9000,    3.0000,   59.0000,    1.8800,    9.6000,    2.9800,	124.5,	133.1,	40,	-0.8]
	letterDict["E"] = [-0.7400,    3.0000,   73.0000,    2.1900,    9.6700,    3.0800,	155.1,	147.13,	62,	-0.77]
	letterDict["F"] = [1.1900,   -2.5000,   91.0000,    2.5800,    9.2400,    5.9100,	203.4,	165.19,	115.5,	1.27]
	letterDict["G"] = [0.4800,         0,    1.0000,    2.3400,    9.6000,    6.0600,	66.4,	75.07,	0,	-0.41]
	letterDict["H"] = [-0.4000,   -0.5000,   82.0000,    1.7800,    8.9700,    7.6400,	167.3,	155.16,	79,	0.49]
	letterDict["I"] = [1.3800,   -1.8000,   57.0000,    2.3200,    9.7600,    6.0400,	168.8,	131.17,	93.5,	1.31]
	letterDict["K"] = [-1.5000,    3.0000,   73.0000,    2.2000,    8.9000,    9.4700,	171.3,	146.19,	100,	-1.18]
	letterDict["L"] = [1.0600,   -1.8000,   57.0000,    2.3600,    9.6000,    6.0400,	167.9,	131.17,	93.5,	1.21]
	letterDict["M"] = [0.6400,   -1.3000,   75.0000,    2.2800,    9.2100,    5.7400,	170.8,	149.21,	94.1,	1.27]
	letterDict["N"] = [-0.7800,    0.2000,   58.0000,    2.1800,    9.0900,   10.7600,	135.2,	132.12,	58.7,	-0.48]
	letterDict["P"] = [0.1200,         0,   42.0000,    1.9900,   10.6000,    6.3000,	129.3,	115.13,	41.9,	0]
	letterDict["Q"] = [-0.8500,    0.2000,   72.0000,    2.1700,    9.1300,    5.6500,	161.1,	146.15,	80.7,	-0.73]
	letterDict["R"] = [-2.5300,    3.0000,  101.0000,    2.1800,    9.0900,   10.7600,	202,	174.2,	105,	-0.84]
	letterDict["S"] = [-0.1800,    0.3000,   31.0000,    2.2100,    9.1500,    5.6800,	99.1,	105.09,	29.3,	-0.5]
	letterDict["T"] = [-0.0500,   -0.4000,   45.0000,    2.1500,    9.1200,    5.6000,	122.1,	119.12,	51.3,	-0.27]	
	letterDict["V"] = [1.0800,   -1.5000,   43.0000,    2.2900,    9.7400,    6.0200,	141.7,	117.15,	71.5,	1.09]
	letterDict["W"] = [0.8100,   -3.4000,  130.0000,    2.3800,    9.3900,    5.8800,	237.6,	204.24,	145.5,	0.88]
	letterDict["Y"] = [0.2600,   -2.3000,  107.0000,    2.2000,    9.1100,    5.6300,	203.6,	181.19,	117.3,	0.33]
	AACategoryLen = 10
	
	probMatr = np.zeros((len(sampleSeq3DArr), 1, len(sampleSeq3DArr[0]), AACategoryLen))
	
	
	sampleNo = 0
	for sequence in sampleSeq3DArr:
	
		AANo	 = 0
		for AA in sequence:
			
			if not AA in letterDict:
				probMatr[sampleNo][0][AANo] = np.full((1,AACategoryLen), 0)
			
			else:
				probMatr[sampleNo][0][AANo]= letterDict[AA]
				
			AANo += 1
		sampleNo += 1
	
	return probMatr


def convertSampleToPhysicsVector_2(sampleSeq3DArr):
	"""
	Convertd the raw data to physico-chemical property
	
	PARAMETER
	---------
	sampleSeq3DArr: 3D numpy array
		X denoted the unknow amino acid.
	
	
	probMatr: Probability Matrix for Samples. Shape (nb_samples, 1, nb_length_of_sequence, nb_AA)
	"""
	
	letterDict = {} 
	letterDict["A"] = [-0.591, -1.302, -0.733, 1.570,-0.146]
	letterDict["C"] = [ -1.343, 0.465, -0.862, -1.020, -0.255]
	letterDict["D"] = [1.050, 0.302, -3.656, -0.259, -3.242]
	letterDict["E"] = [1.357, -1.453, 1.477, 0.113, -0.837]
	letterDict["F"] = [-1.006, -0.590, 1.891, -0.397, 0.412]
	letterDict["G"] = [-0.384, 1.652, 1.330, 1.045, 2.064]
	letterDict["H"] = [0.336, -0.417, -1.673, -1.474, -0.078]
	letterDict["I"] = [-1.239, -0.547, 2.131, 0.393, 0.816]
	letterDict["K"] = [1.831, -0.561, 0.533, -0.277, 1.648]
	letterDict["L"] = [-1.019, -0.987, -1.505, 1.266, -0.912]
	letterDict["M"] = [-0.663, -1.524, 2.219, -1.005, 1.212]
	letterDict["N"] = [0.945, 0.828, 1.299, -0.169, 0.933]
	letterDict["P"] = [0.189, 2.081, -1.628, 0.421, -1.392]
	letterDict["Q"] = [0.931, -0.179, -3.005, -0.503, -1.853]
	letterDict["R"] = [1.538, -0.055, 1.502, 0.440, 2.897]
	letterDict["S"] = [-0.228, 1.399, -4.760, 0.670, -2.647]
	letterDict["T"] = [-0.032, 0.326, 2.213, 0.908, 1.313]	
	letterDict["V"] = [-1.337, -0.279, -0.544, 1.242, -1.262]
	letterDict["W"] = [-0.595, 0.009, 0.672, -2.128, -0.184]
	letterDict["Y"] = [0.260, 0.830, 3.097, -0.838, 1.512]
	AACategoryLen = 5
	
	probMatr = np.zeros((len(sampleSeq3DArr), 1, len(sampleSeq3DArr[0]), AACategoryLen))
	
	
	sampleNo = 0
	for sequence in sampleSeq3DArr:
	
		AANo	 = 0
		for AA in sequence:
			
			if not AA in letterDict:
				probMatr[sampleNo][0][AANo] = np.full((1,AACategoryLen), 0)
			
			else:
				probMatr[sampleNo][0][AANo]= letterDict[AA]
				
			AANo += 1
		sampleNo += 1
	
	return probMatr


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
    elif codingMode == 3:
        probMatr = convertSampleToPhysicsVector(sampleSeq3DArr)
    elif codingMode == 4:
        probMatr = convertSampleToPhysicsVector_2(sampleSeq3DArr)
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
