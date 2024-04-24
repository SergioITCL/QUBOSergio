import numpy as np
from itcl_quantizer.equalizers.adaround.qubocalculation import Input_Round_Calculation, dterm_Calculation, Bterm1_Calculation_Subespacio, Bterm2_Calculation_Subespacio,Bterm_Calculation_Vuelta, Bterm3_Calculation_Subespacio, Bterm4_Calculation_Subespacio,Bterm_Calculation_Vuelta4, Quantum_annealing_simulator,Unir_Diccionarios,Tensor_Redondeo, Matrix_Calculation,QAOA_Solution,Tensor_Redondeo2
kernel = np.random.uniform(-1, 1, size=(31,31))
bias = np.random.uniform(-1, 1, size=(31,))

####
#Obtengo el output a partir de la matriz de pesos el input y el bias
input_f = np.random.uniform(-1, 1, size=(1000,31))






output_f = np.zeros((input_f.shape[0], bias.shape[0]))
for i in range(0,input_f[:,0].shape[0]):
    output_f[i]=kernel.dot(input_f[i])+bias

input_s=0.00392156862745098
kernel_s=0.29605820775032043
bias_s=0.0011610125794130214

xdim=input_f[0].shape
InputDimension=xdim[0]
ydim=bias.shape
OutputDimension=ydim[0]
Numero_datasets=input_f[:,0].shape[0]
Bterm1=np.zeros((Numero_datasets,InputDimension))
Bterm2=np.zeros((Numero_datasets,InputDimension*InputDimension))
Bterm3=np.zeros((Numero_datasets,1))
Bterm4=np.zeros((Numero_datasets,InputDimension))
Redondeo_Kernel=np.zeros((OutputDimension,InputDimension))
Redondeo_Bias=np.zeros((OutputDimension))
Redondeo_Kernel2=np.zeros((OutputDimension,InputDimension))
Redondeo_Bias2=np.zeros((OutputDimension))
M = np.zeros((OutputDimension*InputDimension+OutputDimension, OutputDimension*InputDimension+OutputDimension))
DiccionarioT={}

Input_Round2=Input_Round_Calculation(input_f,input_s)
dTermino2=dterm_Calculation(bias,kernel,Input_Round2,output_f,kernel_s,input_s,bias_s)
#print(dTermino2)
for l in range(0,1):
    print('l',l)
    Dicionario1={}
    Dicionario2={}
    diccionario2={}
    Dicionario3={}
    Dicionario4={}
    diccionario4={}
    Dicionario={}
    diccionario1=Bterm1_Calculation_Subespacio(Dicionario1,Bterm1,Input_Round2,dTermino2,l,InputDimension,Numero_datasets,kernel_s,input_s)
    #print(diccionario11)
    if l==0:
        diccionario2,Bt2=Bterm2_Calculation_Subespacio(Dicionario2,Bterm2,Input_Round2,l,InputDimension,Numero_datasets,kernel_s,input_s)
    else:
        diccionario2=Bterm_Calculation_Vuelta(Bt2,diccionario2,InputDimension,l)
    diccionario3=Bterm3_Calculation_Subespacio(Dicionario3,Bterm3,dTermino2,l,InputDimension,OutputDimension,bias_s,input_s,kernel_s,Numero_datasets)
    #print(diccionario3)
    if l==0:
        diccionario4,Bt4=Bterm4_Calculation_Subespacio(Dicionario4,Bterm4,Input_Round2,l,InputDimension,OutputDimension,bias_s,input_s,kernel_s,Numero_datasets)
    else:
        diccionario4=Bterm_Calculation_Vuelta4(Dicionario4,Bt4,InputDimension,OutputDimension,l)
    #print(diccionario4)
    diccionarioM=Unir_Diccionarios(diccionario1,diccionario2,diccionario3,diccionario4,Dicionario)
    print('diccionario1',diccionarioM)
    DiccionarioT=Unir_Diccionarios(diccionario1,diccionario2,diccionario3,diccionario4,DiccionarioT)
    Matrix_Calculation(M,diccionario1,diccionario2,diccionario3,diccionario4, OutputDimension,InputDimension)
    result=Quantum_annealing_simulator(diccionarioM,l)
    #print('result1',result)
    #result2=QAOA_Solution(diccionario)
    Redondeo_Kernel[l], Redondeo_Bias[l]=Tensor_Redondeo(result,Redondeo_Kernel,Redondeo_Bias,l,InputDimension)
#print('diccionario1',DiccionarioT)     
#Redondeo_Kernel = np.random.choice([0, 1], size=(OutputDimension, InputDimension))
#Redondeo_Bias = np.random.choice([0, 1], size=(OutputDimension))

tensors=[Redondeo_Kernel,Redondeo_Bias]



#print('cuant',Cuant)
#print('bias',biasCuant)



'''
M = np.zeros((OutputDimension*InputDimension+OutputDimension, OutputDimension*InputDimension+OutputDimension))
for i in range(0,OutputDimension*InputDimension+OutputDimension):
    for j in range(0,OutputDimension*InputDimension+OutputDimension):
        if (i==j) and ((i+1,j+1) in diccionario1):
            M[i][j]=diccionario1[(i+1,j+1)]
        if (i+1,j+1) in diccionario2:
                M[i][j]=diccionario2[(i+1,j+1)]
        if (i==j) and ((i+1,j+1) in diccionario3):
            M[i][j]=diccionario3[(i+1,j+1)]  
        if (i+1,j+1) in diccionario4:
                M[i][j]=diccionario4[(i+1,j+1)]'''


'''
def test_Input_Round_Calculation():
    prueba1=Input_Round_Calculation(input_f[0],input_s,InputDimension)
    prueba2=round(input_f[0][1]/input_s)
    assert prueba1[1]==prueba2

def test_dTermino():
    prueba1=dTermino=dterm_Calculation(bias,kernel,Input_Round,output_f,kernel_s,input_s,bias_s,InputDimension,OutputDimension,s)
'''

    
                 

