# The required utilities

import numpy as np
np.set_printoptions(precision = 2)

def calc_table(C_in, C_print_flag=0):
    if(np.shape(C_in)==(8,8)):
        Cs5 = np.zeros((3,3))
        Cs5[1:,1:] = C_in[-4:,-4:]
        Cs5[0,1:] = np.sum(C_in[:3,3:], 0)
        Cs5[1:,0] = np.sum(C_in[3:,:3], 1)
        Cs5[0,0] = np.sum(C_in[:3,:3])
    else:
        Cs5=C_in

    if(C_print_flag==1):
        print(Cs5)
    if(C_print_flag==2):
        print(C_in)

    ##################################################
    selected_class = 0
    TpV = Cs5[selected_class,selected_class]
    FpV = np.sum(Cs5[selected_class,:])
    accV = np.round(100*(TpV)/(FpV),2)
    
    selected_class = 1
    TpV = Cs5[selected_class,selected_class]
    FpV = np.sum(Cs5[selected_class,:])
    senV = np.round(100*(TpV)/(FpV),2)
    
    selected_class = 2
    TpV = Cs5[selected_class,selected_class]
    FpV = np.sum(Cs5[selected_class,:])
    speV = np.round(100*(TpV)/(FpV),2)
    

    ##################################################

    outputMat = np.reshape(np.asarray([accV, senV, speV]), (1,-1))
    
    return outputMat


def calc_tables(allCs, n_classes): #allCs = C #C_in = C
    Cs=np.zeros((n_classes,n_classes))
    Cs = Cs + allCs
    tab = calc_table(Cs)

    return tab

#n_classes = 3
#C_in = np.array([[521,0,0,0,65,0,780,4],[0,1,0,1,0,0,0,2],[0,0,9,2,1,0,0,5],[0,0,0,775,0,0,0,2],[0,0,0,0,421,0,0,0],[0,0,0,0,0,722,0,1],[3,0,0,0,0,0,1351,0],[0,0,0,0,0,0,7,609]])
#calc_tables(C_in, 8)
