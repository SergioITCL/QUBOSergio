import numpy as np
from itcl_quantizer.equalizers.adaround.qubocalculation import Input_Round_Calculation, dterm_Calculation, Bterm1_Calculation_Subespacio, Bterm2_Calculation_Subespacio, Bterm3_Calculation_Subespacio, Bterm4_Calculation_Subespacio, Quantum_annealing_simulator,Unir_Diccionarios,Tensor_Redondeo, Matrix_Calculation, QAOA_Solution,Tensor_Redondeo2


kernel = np.random.uniform(-1, 1, size=(20,20))
bias = np.random.uniform(-1, 1, size=(20,))

####
#Obtengo el output a partir de la matriz de pesos el input y el bias
input_f = np.random.uniform(-1, 1, size=(1000,20))






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
Cuant=np.zeros((ydim[0],xdim[0]))
biasCuant=np.zeros((ydim[0]))
Cuant2=np.zeros((ydim[0],xdim[0]))
biasCuant2=np.zeros((ydim[0]))
Bterm1=np.zeros((InputDimension))
Bterm2=np.zeros((InputDimension*InputDimension))
Bterm3=np.zeros(1)
Bterm4=np.zeros(InputDimension)
Redondeo_Kernel=np.zeros((OutputDimension,InputDimension))
Redondeo_Bias=np.zeros((OutputDimension))
Redondeo_Kernel2=np.zeros((OutputDimension,InputDimension))
Redondeo_Bias2=np.zeros((OutputDimension))
M = np.zeros((OutputDimension*InputDimension+OutputDimension, OutputDimension*InputDimension+OutputDimension))
DiccionarioT={}
for l in range(0,1):
    Dicionario1={}
    Dicionario2={}
    Dicionario3={}
    Dicionario4={}
    Dicionario={}
    for s in range(0,Numero_datasets):
        Input_Round=Input_Round_Calculation(input_f[s],input_s,InputDimension)
        #print('Input',Input_Round)
        dTermino=dterm_Calculation(bias,kernel,Input_Round,output_f,kernel_s,input_s,bias_s,InputDimension,OutputDimension,s)
        #print('dterm',dTermino)
        diccionario1=Bterm1_Calculation_Subespacio(Dicionario1,Bterm1,Input_Round,dTermino,l,s,InputDimension,Numero_datasets,kernel_s,input_s)
        diccionario2=Bterm2_Calculation_Subespacio(Dicionario2,Bterm2,Input_Round,l,InputDimension,OutputDimension,Numero_datasets,kernel_s,input_s)
        diccionario3=Bterm3_Calculation_Subespacio(Dicionario3,Bterm3,dTermino,l,InputDimension,OutputDimension,bias_s,input_s,kernel_s,Numero_datasets)
        diccionario4=Bterm4_Calculation_Subespacio(Dicionario4,Bterm4,Input_Round,l,InputDimension,OutputDimension,bias_s,input_s,kernel_s,Numero_datasets)
        diccionario=Unir_Diccionarios(diccionario1,diccionario2,diccionario3,diccionario4,Dicionario)
        DiccionarioT=Unir_Diccionarios(diccionario1,diccionario2,diccionario3,diccionario4,DiccionarioT)
    print(diccionario)
    print('l',l)
    Matrix_Calculation(M,diccionario1,diccionario2,diccionario3,diccionario4, OutputDimension,InputDimension)
    result=Quantum_annealing_simulator(diccionario,l)
    #result2=QAOA_Solution(diccionario)
    Redondeo_Kernel[l], Redondeo_Bias[l]=Tensor_Redondeo(result,Cuant,biasCuant,l,InputDimension)
    #Redondeo_Kernel2[l], Redondeo_Bias2[l]=Tensor_Redondeo2(result2,Cuant2,biasCuant2,l,InputDimension)
#print(DiccionarioT)
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

    
                 

