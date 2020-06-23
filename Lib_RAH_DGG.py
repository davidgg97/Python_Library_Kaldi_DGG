#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jun  8 14:42:24 2020

@author: davidgomezgomez
"""

import os
import numpy as np
import pandas as pd
import math
import seaborn as sn
import matplotlib.pyplot as plt
from matplotlib import colors
from scipy.stats import norm
from matplotlib import cm
from matplotlib.colors import ListedColormap



##### CONVERSION DE ARCHIVO .TXT A DATAFRAME PARA FACILITAR EL TRABAJO EN PYTHON #####

#Sirve para establecer en qué carpeta se está trabajando (ruta)
working_path = os.getcwd() 

#Se construye un dataframe importando el archivo txt separado por tabuladores
texto_kaldi = pd.read_csv("salida.txt",delimiter="\t") 

#Se cambia (acorta) el de la columna que indica el nº de sílabas para poder trabajar con ella más facilmente
texto_kaldi.rename(columns={'Number of Syllables': 'NSyllables'}, inplace = True) 





##### CALCULO DE EL PORCENTAJE DE ACIERTO SEGUN EL NUMERO DE SILABAS DE LA PALABRA #####

def PorcentajeAciertos(df):
    #Porcentaje de palabras de 2 sílabas
    two_syl = round((len(df[df.NSyllables == 2])/ len(df))*100, 2)
    #Porcentaje de palabras de 3 sílabas
    three_syl = round((len(df[df.NSyllables == 3])/ len(df))*100, 2) 
    #Porcentaje de palabras de 4 sílabas
    four_syl = round((len(df[df.NSyllables == 4])/ len(df))*100, 2) 
    
    
    #Calcula el % de total, crea un array en el q solo se queda con los elementos q df.prueba == 1 y lo divide por el total, redondeo a dos decimales
    Corr_Total = round(((len(df[df.Correct == 1])/ len(df)) * 100), 2) 
    #Calcula el % de acierto para palabras de 2 sílabas, redondeamos a dos decimales
    Corr_two = round(((len(df[(df.Correct == 1) & (df.NSyllables == 2)])/ len(df[df.NSyllables == 2])) * 100), 2) 
    #Calcula el % de acierto para palabras de 3 sílabas, redondeamos a dos decimales
    Corr_three = round(((len(df[(df.Correct == 1) & (df.NSyllables == 3)])/ len(df[df.NSyllables == 3])) * 100), 2)
    #Calcula el % de acierto para palabras de 4 sílabas, redondeamos a dos decimales
    Corr_four = round(((len(df[(df.Correct == 1) & (df.NSyllables == 4)])/ len(df[df.NSyllables == 4])) * 100), 2) 
    
    #Imprime los resultados de la funcion
    print("El porcentaje total correcto es:", Corr_Total , "% \n")
    print("El porcentaje correcto con 2 sílabas es:", Corr_two, "% ", "Con un porcentaje de palabras total de:", two_syl, "% \n")
    print("El porcentaje correcto con 3 sílabas es:", Corr_three, "% ", "Con un porcentaje de palabras total de:", three_syl, "% \n")
    print("El porcentaje correcto con 4 sílabas es:", Corr_four, "% ", "Con un porcentaje de palabras total de:", four_syl, "%""\n")
    
    



##### CALCULO DE LA MATRIZ DE CONFUSION TOTALES PARA VOCALES Y CONSONANTES Y VISUALIZACION DE ESTAS MATRICES #####

#Matriz de confusión para VOCALES
def ConfusionMatrixVoc(df):
    conf_matV = pd.crosstab(df['TargetV'], df['RespV'], rownames=['Target'], colnames=['Response'], margins = True); 
    #Elimina la fila All y *(no da información relevante) Vocales
    ncmV = conf_matV.drop(["All", "*"], axis = 0);
    #Obtiene la matriz de confusion total - Elimina la columna All y * (no da información relevante) Vocales
    confusion_matrixV = ncmV.drop(["All", "*"], axis = 1); 
    return confusion_matrixV

#Matriz de confusión Normalizada para VOCALES
def NormalConfusionMatrixVoc(df):
    conf_matV = pd.crosstab(df['TargetV'], df['RespV'], rownames=['Target'], colnames=['Response'], margins = True); 
    #Normalizacion de la matriz de confusion Vocales
    ncmV1 = conf_matV/conf_matV.max().astype(np.float64); 
    #Elimina la fila All y * (no da información relevante) Vocales
    ncmV2 = ncmV1.drop(["All", "*"], axis = 0); 
    #Elimina la columna All y *(no nos da información) Vocales
    ncmV3 = ncmV2.drop(["All", "*"], axis = 1); 
    #Obtiene la Matriz de Confusión Normalizada - Redondea los datos a dos decimales en df Vocales
    normalize_confusion_matrixV = ncmV3.round(2) 
    return normalize_confusion_matrixV

#Matriz de confusión para CONSONANTES
def ConfusionMatrixCons(df):
    conf_matC = pd.crosstab(df['TargetC'], df['RespC'], rownames=['Target'], colnames=['Response'], margins = True); 
    #Elimina la fila All y ** (no da información relevante) Consonantes
    ncmC = conf_matC.drop(["All", "**"], axis = 0); 
    #Obtiene matriz de confusion total - Elimina las columnas  All, **, gr y zr (no da informacion relevante) Consonantes
    confusion_matrixC = ncmC.drop(["All", "**","gr", "zr"], axis = 1); 
    return confusion_matrixC

#Matriz de confusión Normalizada para CONSONANTES
def NormalConfusionMatrixCons(df):
    conf_matC = pd.crosstab(df['TargetC'], df['RespC'], rownames=['Target'], colnames=['Response'], margins = True); 
    #Normalizacion de matriz de confusion Consonantes
    ncmC1 = conf_matC/conf_matC.max().astype(np.float64); 
    #Eliminacion de la fila  All y ** (no da información relevante) Consonantes
    ncmC2 = ncmC1.drop(["All","**" ], axis = 0); 
    #Eliminacion las columnas All, **, gr y zr (no da información relevante) Consonantes
    ncmC3 = ncmC2.drop(["All", "**","gr", "zr"], axis = 1); 
    #Obtencion Matriz de Confusion Normalizada - Redondeo de los datos a dos decimales en df Consonantes
    normarlize_confusion_matrixC = ncmC3.round(2) 
    return normarlize_confusion_matrixC





##### AGRUPACIÓN DE CONSONANTES POR CRITERIO DE SONORIDAD, LUGAR DE ARTICULACION Y MODO DE ARTICULACION #####

# 1. SONORIDAD: Sordas o Sonoras #
def CSonora(df):
    #Creacion de un dataframe con las consonantes Sonoras
    ncmC1 = ConfusionMatrixCons(df)
    CSonora1 = ncmC1.filter(["b","d","g","l","m","n","ny","r","rr","y"], axis=1)
    Sonoridad_Sonora = CSonora1.filter(["b","d","g","l","m","n","ny","r","rr","y"], axis=0) 
    return Sonoridad_Sonora

def CSorda(df):
    #Creacion de un dataframe con las consonantes Sonoras
    ncmC1 = ConfusionMatrixCons(df)
    CSorda1 = ncmC1.filter(["ch","f","k","p","s","t","x","z"], axis=1)
    Sonoridad_Sorda = CSorda1.filter(["ch","f","k","p","s","t","x","z"], axis=0) 
    return Sonoridad_Sorda   
 
# 1.1 Sonoridad Normalizadas
def NormalizeCSonora(df):
    #Creacion de un dataframe con las consonantes Sonoras normalizadas
    ncmC1 = NormalConfusionMatrixCons(df)
    CSonora1 = ncmC1.filter(["b","d","g","l","m","n","ny","r","rr","y"], axis=1)
    Sonoridad_Sonora = CSonora1.filter(["b","d","g","l","m","n","ny","r","rr","y"], axis=0) 
    return Sonoridad_Sonora

def NormalizeCSorda(df):
    #Creaacion de un dataframe con las consonantes Sonoras normalizadas
    ncmC1 = NormalConfusionMatrixCons(df)
    CSorda1 = ncmC1.filter(["ch","f","k","p","s","t","x","z"], axis=1)
    Sonoridad_Sorda = CSorda1.filter(["ch","f","k","p","s","t","x","z"], axis=0) 
    return Sonoridad_Sorda


# 2. LUGAR DE ARTICULACION: Frontal, Coronal o Back #
def CFrontal(df):
    #Creacion de un dataframe con las consonantes Frontal
    ncmC1 = ConfusionMatrixCons(df)
    CFrontal1 = ncmC1.filter(["b","f","m","p"], axis=1)
    Lugar_Frontal = CFrontal1.filter(["b","f","m","p"], axis=0) 
    return Lugar_Frontal

def CCoronal(df):
    #Creacion de un dataframe con las consonantes Coronal
    ncmC1 = ConfusionMatrixCons(df)
    CCoronal1 = ncmC1.filter(["ch","d","l","n","ny","r","rr","s","t","y","z"], axis=1)
    Lugar_Coronal = CCoronal1.filter(["ch","d","l","n","ny","r","rr","s","t","y","z"], axis=0) 
    return Lugar_Coronal   

def CBack(df):
    #Creacion de un dataframe con las consonantes Back
    ncmC1 = ConfusionMatrixCons(df)
    CBack1 = ncmC1.filter(["g","k","x"], axis=1)
    Lugar_Back = CBack1.filter(["g","k","x"], axis=0) 
    return Lugar_Back 

# 2.1 Lugar de articulacion Normalizadas
def NormalizeCFrontal(df):
    #Creacion de un dataframe con las consonantes Frontal normalizadas
    ncmC1 = NormalConfusionMatrixCons(df)
    CFrontal1 = ncmC1.filter(["b","f","m","p"], axis=1)
    Lugar_Frontal = CFrontal1.filter(["b","f","m","p"], axis=0) 
    return Lugar_Frontal

def NormalizeCCoronal(df):
    #Creacion de un dataframe con las consonantes Coronal normalizadas
    ncmC1 = NormalConfusionMatrixCons(df)
    CCoronal1 = ncmC1.filter(["ch","d","l","n","ny","r","rr","s","t","y","z"], axis=1)
    Lugar_Coronal = CCoronal1.filter(["ch","d","l","n","ny","r","rr","s","t","y","z"], axis=0) 
    return Lugar_Coronal

def NormalizeCBack(df):
    #Creacion de un dataframe con las consonantes Back normalizadas
    ncmC1 = NormalConfusionMatrixCons(df)
    CBack1 = ncmC1.filter(["g","k","x"], axis=1)
    Lugar_Back = CBack1.filter(["g","k","x"], axis=0) 
    return Lugar_Back


# 3. MODO DE ARTICULACION: Oclusiva, Africada, Fricativa, Nasal, Aproximantes #
def COclusiva(df):
    #Creacion de un dataframe con las consonantes Oclusivas
    ncmC1 = ConfusionMatrixCons(df)
    COclusiva1 = ncmC1.filter(["b","d","g","k","p","t"], axis=1)
    Modo_Oclusiva = COclusiva1.filter(["b","d","g","k","p","t"], axis=0) 
    return Modo_Oclusiva

def CAfricada(df):
    #Creacion de un dataframe con las consonantes Africadas
    ncmC1 = ConfusionMatrixCons(df)
    CAfricada1 = ncmC1.filter(["ch"], axis=1)
    Modo_Africada = CAfricada1.filter(["ch"], axis=0) 
    return Modo_Africada

def CFricativa(df):
    #Creacion de un dataframe con las consonantes Fricativa
    ncmC1 = ConfusionMatrixCons(df)
    CFricativa1 = ncmC1.filter(["f","s","x","y","z"], axis=1)
    Modo_Fricativa = CFricativa1.filter(["f","s","x","y","z"], axis=0) 
    return Modo_Fricativa

def CNasal(df):
    #Creacion de un dataframe con las consonantes Nasal
    ncmC1 = ConfusionMatrixCons(df)
    CNasal1 = ncmC1.filter(["m","n","ny"], axis=1)
    Modo_Nasal = CNasal1.filter(["m","n","ny"], axis=0) 
    return Modo_Nasal

def CAproximante(df):
    #Creacion de un dataframe con las consonantes Aproximante
    ncmC1 = ConfusionMatrixCons(df)
    CAproximante1 = ncmC1.filter(["l","r","rr"], axis=1)
    Modo_Aproximante = CAproximante1.filter(["l","r","rr"], axis=0) 
    return Modo_Aproximante

# 2.1 Modo de articulacion Normalizadas
def NormalizeCOclusiva(df):
    #Creacion de un dataframe con las consonantes Oclusivas normalizadas
    ncmC1 = NormalConfusionMatrixCons(df)
    COclusiva1 = ncmC1.filter(["b","d","g","k","p","t"], axis=1)
    Modo_Oclusiva = COclusiva1.filter(["b","d","g","k","p","t"], axis=0) 
    return Modo_Oclusiva

def NormalizeCAfricada(df):
    #Creacion de un dataframe con las consonantes Africadas normalizadas
    ncmC1 = NormalConfusionMatrixCons(df)
    CAfricada1 = ncmC1.filter(["ch"], axis=1)
    Modo_Africada = CAfricada1.filter(["ch"], axis=0) 
    return Modo_Africada

def NormalizeCFricativa(df):
    #Creacion de un dataframe con las consonantes Fricativa normalizadas
    ncmC1 = NormalConfusionMatrixCons(df)
    CFricativa1 = ncmC1.filter(["f","s","x","y","z"], axis=1)
    Modo_Fricativa = CFricativa1.filter(["f","s","x","y","z"], axis=0) 
    return Modo_Fricativa

def NormalizeCNasal(df):
    #Creacion de un dataframe con las consonantes Nasal normalizadas
    ncmC1 = NormalConfusionMatrixCons(df)
    CNasal1 = ncmC1.filter(["m","n","ny"], axis=1)
    Modo_Nasal = CNasal1.filter(["m","n","ny"], axis=0) 
    return Modo_Nasal

def NormalizeCAproximante(df):
    #Creacion de un dataframe con las consonantes Aproximante normalizadas
    ncmC1 = NormalConfusionMatrixCons(df)
    CAproximante1 = ncmC1.filter(["l","r","rr"], axis=1)
    Modo_Aproximante = CAproximante1.filter(["l","r","rr"], axis=0) 
    return Modo_Aproximante





##### IMPRIMIR MATRICES #####

#Funcion para imprimir las matrices de confusión normalizadas, utilizando un mapa de calor con diferentes gradaciones de color
#Gradaciones de color diferentes y con una escala logarítmica divididas en intervalos


#Escala de color Naranja - Azul
def PlotMatOrangeBlue(normalize_confusion_matrix, nombre):
    
    #Se define el color, en una escala azul-naranja
    topO = cm.get_cmap('Oranges_r', 128) #En una escala de colores, el naranja representa la mitad 128/256
    bottomB = cm.get_cmap('Blues', 128) #En una escala de colores, el naranja representa 128/256
    newcolorsOB = np.vstack((topO(np.linspace(0, 1, 128)),
                       bottomB(np.linspace(0, 1, 128))))
    newcmpOB = ListedColormap(newcolorsOB, name='OrangeBlue')

    if len(normalize_confusion_matrix) >= 7: 
        plt.figure(figsize=(15, 10))
        #Mapa de calor, dependiendo la escala de colores que se quiera (escala creada u otra ya existente),
        #se asigna la escala a cmap; en este caso, cmap = newcmpOB (escala Orange y Blue)
        sn.heatmap(normalize_confusion_matrix, cmap= newcmpOB, annot_kws={"size": 13},
           norm=colors.LogNorm(vmin=0.01,vmax=1), vmin=0.01, vmax=1, annot=True);
        plt.title("Mapa de calor Matriz de confusion - " + nombre)
        plt.tight_layout() #Ajusta la figura (funciona también sin este parametro)
        plt.grid(which='minor', alpha=0.1) #Para hacer el grid de un tono más claro y no tape datos
        plt.grid(which='major', alpha=0.2) #Para hacer el grid de un tono más claro y no tape datos
        plt.savefig(nombre) #Guarda la figura creada en la carpeta donde se encuentre el fichero de ejecución
        
    else:
        plt.figure(figsize=(8, 4))
        #Mapa de calor, dependiendo la escala de colores que se quiera (escala creada u otra ya existente),
        #se asigna la escala a cmap; en este caso, cmap = newcmpOB (escala Orange y Blue)
        sn.heatmap(normalize_confusion_matrix, cmap= newcmpOB, annot_kws={"size": 20},
           norm=colors.LogNorm(vmin=0.01,vmax=1), vmin=0.01, vmax=1, annot=True);
        plt.title("Mapa de calor Matriz de confusion - " + nombre)
        plt.tight_layout() #Ajusta la figura (funciona también sin este parametro)
        plt.grid(which='minor', alpha=0.1) #Para hacer el grid de un tono más claro y no tape datos
        plt.grid(which='major', alpha=0.2) #Para hacer el grid de un tono más claro y no tape datos
        plt.savefig(nombre) #Guarda la figura creada en la carpeta donde se encuentre el fichero de ejecución
        
    return plt.show()
    

#Escala de color Viridis 
def PlotMatViridis(normalize_confusion_matrix, nombre):
    
    #Se define el color Viridis con una escala de 256 datos
    viridis = cm.get_cmap('viridis', 256)

    if len(normalize_confusion_matrix) >= 7: 
        plt.figure(figsize=(15, 10))
        #Mapa de calor, dependiendo la escala de colores que se quiera (escala creada u otra ya existente),
        #se asigna la escala a cmap; en este caso, cmap = newcmpOB (escala Orange y Blue)
        sn.heatmap(normalize_confusion_matrix, cmap= viridis, annot_kws={"size": 13},
           norm=colors.LogNorm(vmin=0.01,vmax=1), vmin=0.01, vmax=1, annot=True);
        plt.title("Mapa de calor Matriz de confusion - " + nombre)
        plt.tight_layout() #Ajusta la figura (funciona también sin este parametro)
        plt.grid(which='minor', alpha=0.1) #Para hacer el grid de un tono más claro y no tape datos
        plt.grid(which='major', alpha=0.2) #Para hacer el grid de un tono más claro y no tape datos
        plt.savefig(nombre) #Guarda la figura creada en la carpeta donde se encuentre el fichero de ejecución
        
    else:
        plt.figure(figsize=(8, 4))
        #Mapa de calor, dependiendo la escala de colores que se quiera (escala creada u otra ya existente),
        #se asigna la escala a cmap; en este caso, cmap = newcmpOB (escala Orange y Blue)
        sn.heatmap(normalize_confusion_matrix, cmap= viridis, annot_kws={"size": 20},
           norm=colors.LogNorm(vmin=0.01,vmax=1), vmin=0.01, vmax=1, annot=True);
        plt.title("Mapa de calor Matriz de confusion - " + nombre)
        plt.tight_layout() #Ajusta la figura (funciona también sin este parametro)
        plt.grid(which='minor', alpha=0.1) #Para hacer el grid de un tono más claro y no tape datos
        plt.grid(which='major', alpha=0.2) #Para hacer el grid de un tono más claro y no tape datos
        plt.savefig(nombre) #Guarda la figura creada en la carpeta donde se encuentre el fichero de ejecución
    
    return plt.show()





##### CREACION DE FUNCIONES ESTADISTICAS #####
    
# Parámetros que correspondan, medidos en tanto por uno en vez de tanto por ciento

#Definicion de la funcion que calcula la sensibilidad de la muestra/matriz 2x2
def sensibilidad(VP, FN, FP, VN):
    #Calculo de la sensibilidad, redondeo 3 decimales
    resultado_sen = round((VP)/(VP + FN), 3); 
    return (resultado_sen)

#Definicion de la funcion que calcula la especificidad de la muestra/matriz 2x2
def especificidad(VP, FN, FP, VN): 
    #Calculo de la especificidad, redondeo 3 decimales
    resultado_espec = round((VN)/(VN + FP), 3); 
    return (resultado_espec)

#Definicion de la funcion que calcula la exactitud de la muestra/matriz 2x2
def exactitud(VP, FN, FP, VN):
    #Calculo de la exactitud, redondeo 3 decimales
    resultado_exact = round((VP + VN)/(VP + FN + FP + VN), 3); 
    return (resultado_exact)

#Definicion de la funcion que calcula la precision de la muestra/matriz 2x2
def precision(VP, FN, FP, VN):
    #Calculo de la especificidad, redondeo 3 decimales
    resultado_precis = round((VP)/(VP + FP), 3); 
    return (resultado_precis)

#Definicion de la funcion Misclassification Ratio or Mean error (MRC) - Error Medio
def errorMedio (VP, FN, FP, VN):
    #Formula error medio, redondeo 3 decimales
    MCR = round((FP + FN) / (VP + FN + FP + VN), 3); 
    return (MCR)

#Definicion del F-score (Valor F) = Precision x Sensibilidad
def valorF(VP, FN, FP, VN):
    #Calculo del valor-F, redondeo 3 decimales
    resultado_valorF = round(2*(sensibilidad(VP, FN, FP, VN) * precision(VP, FN, FP, VN))/
                               (sensibilidad(VP, FN, FP, VN) + precision(VP, FN, FP, VN)), 3);
    return (resultado_valorF)

#Definicion del ratio de falsos positivos - false positive rate (FPR)
def falsePositive(VP, FN, FP, VN):
    #Calculo FPR, rendondeo 3 decimales
    resultado_FPR = round((FP)/(FP + VN), 3); 
    return (resultado_FPR)

#Definicion de la funcion que calcula el Coeficiente de Matthew
def matthew(VP,FN, FP, VN):
    #Formula Coeficiente Matthew, redondeo a 3 decimales
    resultado_matthew = round((((VP * VN) - (FP * FN)) / (math.sqrt((VP + FP) * (VP + FN) * (VN + FP) * (VN + FN)))), 3); 
    return (resultado_matthew)

#Definicion de la funcion que calcula el Indice de Jaccard 
def jaccard(VP, FN, FP, VN):
    #Formula Indice Jaccard, redondeo 3 decimales
    resultado_jaccard = round((VP) / (VP + FP + FN), 3); 
    return (resultado_jaccard)

#Definimos la funcion que calcula el parametro d'
def parametroD(VP, FN, FP, VN):
    #Definicion de funcion para calcular el Parametro D', redondeo a 3 decimales
    H = VP/(VP+FN) #Probabilidad condicional de acierto
    FA = FP/(FP+VN) #Probabilidad condicional de fallo
    zH = norm.ppf(H) #Funcion normal de distribucion inversa para la probabilidad de acierto
    zFA = norm.ppf(FA)#Funcion normal de distribucion inversa para la probabilidad de fallo
    parametro_d = zH -zFA #Calculo del parametro d'
    return (round(parametro_d, 3))





##### TRANSFORMACION  A MATRICES 2X2 #####

#VOCALES
def Matriz2x2(confusion_matrix):
    #Definicion de qué es un verdadero positivo, falso positivo, falso negativo y verdadero negativo 
    verdaderos_positivos = np.diag(confusion_matrix) #Definicion de verdaderos positivos de la matriz 
    falsos_positivos = confusion_matrix.sum(axis=0) - verdaderos_positivos #Definicion de los falsos positivos de la matriz 
    falsos_negativos = confusion_matrix.sum(axis=1) - verdaderos_positivos #Definicion de los falsos negativos de la matriz 
    verdaderos_negativos = (confusion_matrix.to_numpy().sum() - (verdaderos_positivos + falsos_positivos + falsos_negativos)) #Definicion de los verdaderos negativos de la matriz 
    #Lista con Dataframe de matrices de confusion 2x2 p
    Matrices2x2 = [] #Creacion de una lista (de dataframes) en la que se guardan la matriz de confusion de cada letra
    for dato, vp, fp, vn, fn in zip(
            confusion_matrix.columns,
            verdaderos_positivos, falsos_positivos,
            verdaderos_negativos, falsos_negativos):
        
        #Creacion del Dataframe de la matriz de confusion 2x2 para cada letra,
        #cuenta con una tercera fila con el nombre de la letra, para poder identificar el df
        frame = pd.DataFrame.from_dict(
            {"Positive(1)": (vp, fp), "Negative(0)": (fn, vn), "Name": (dato, "-")}, 
            orient="Index",
            columns=("Positive(1)", "Negative(0)")
            )
        frame.index.name = f'Predicted Values for "{dato}"'
        frame.columns.name = f'Actual values for "{dato}"'
        Matrices2x2.append(frame) #Se va rellenando la lista con los dataframes obtenidos
    return Matrices2x2





##### APLICACON DE PARÁMETROS ESTADÍSTICO A VOCALES Y CONSONANTES #####
    
#Calculos de las funciones estadisticas previamente definidas 
def FuncionesEstadisticas(confusion_matrix):
    verdaderos_positivos = np.diag(confusion_matrix) 
    falsos_positivos = confusion_matrix.sum(axis=0) - verdaderos_positivos 
    falsos_negativos = confusion_matrix.sum(axis=1) - verdaderos_positivos 
    verdaderos_negativos = (confusion_matrix.to_numpy().sum() - (verdaderos_positivos + falsos_positivos + falsos_negativos)) 
    #Lista con Dataframe de matrices de confusion 2x2 
    Matrices2x2 = [] 
    for dato, vp, fp, vn, fn in zip(
            confusion_matrix.columns,
            verdaderos_positivos, falsos_positivos,
            verdaderos_negativos, falsos_negativos):

        #Creacion del Dataframe de la matriz de confusion 2x2 para cada letra,
        # uenta con una tercera fila con el nombre de la letra, para poder identificar el df
        frame = pd.DataFrame.from_dict(
            {"Positive(1)": (vp, fp), "Negative(0)": (fn, vn), "Name": (dato, "-")}, 
            orient="Index",
            columns=("Positive(1)", "Negative(0)")
            )
        frame.index.name = f'Predicted Values for "{dato}"'
        frame.columns.name = f'Actual values for "{dato}"'
        Matrices2x2.append(frame) 

    #Creacion de listas para almacenar los calculos de los datos
    #Creacion de una lista para almacenar los cálculos de la sensibilidad 
    lista_sensibilidad = [] 
    #Creacion de una lista para almacenar los datos de la especificidad de cada una de las letras
    lista_Especificidad = [] 
    #Creacion de una lista para almacenar los datos de la exactitud de cada una de las letras
    lista_Exactitud = [] 
    #Creacion de una lista para almacenar los datos de la precison de cada una de las letras
    lista_Precision = [] 
    #Creacion de una lista para almacenar los cálculos del Coeficiente de Matthew 
    lista_Matthews = [] 
    #Creacion de una lista para almacenar los cálculos del Indice de Jaccard
    lista_Jaccard = []  
    #Creacion de una lista para almacenar los cálculos del parametro d'
    lista_ParametroD = []  
    #Creacion de una lista para almacenar los cálculos del error medio cada una de las letras
    lista_ErrorMedio = [] 
    #Creacion de una lista para almacenar los cálculos del valor F cada una de las letras
    lista_ValorF = [] 
    #Creacion de una lista para almacenar los calulos del ratio de falsos positivos para letras
    lista_FPR = [] 
    #Creacion de una lista, que va a contener los nombres de las letras, cuyas posiciones van a ser las mismas que ocupan en las demas listas
    #Es decir, si se quiere saber qué letra es la posicion 4 de una lista estadística, se va a la lista nombre, se mira la posición 4 y esa será la letra correspondiente   
    lista_Letra_Posicion = [] 
                            
    #Bucle para calcular los Parametros Estadisticos para cada letra
    for i in range(0,len(Matrices2x2)):
        VP = Matrices2x2[i]['Positive(1)'][0] #Definicion de la posición en la que se encuentra el Verdadero Positivo
        FN = Matrices2x2[i]['Positive(1)'][1] #Definicion de la posición en la que se encuentra el Verdadero Positivo
        FP = Matrices2x2[i]['Negative(0)'][0] #Definicion de la posición en la que se encuentra el Verdadero Positivo
        VN = Matrices2x2[i]['Negative(0)'][1] #Definicion de la posición en la que se encuentra el Verdadero Positivo
        Nombre_Letra = Matrices2x2[i]['Positive(1)'][2] #Sabe el nombre de la letra
    
        sensibilidadf = sensibilidad(VP, FN, FP, VN) #Calculo de la sensibilidad 
        especificidadf = especificidad(VP, FN, FP, VN) #Calculo de la especificidad 
        exactitudf = exactitud(VP, FN, FP, VN) #Calculo de la exactitud
        precisionf = precision(VP, FN, FP, VN) #Calculo de la precision 
        matthewf = matthew(VP, FN, FP, VN) #Calculo del Coeficiente de Matthew 
        jaccardf = jaccard(VP, FN, FP, VN)#Calculo del Indice de Jaccard  
        parametroDf = parametroD(VP, FN, FP, VN)#Calculo deL Parametro d'
        errorMedf = errorMedio(VP, FN, FP, VN)#Calculo del error medio para las Consonantes
        valorFf = valorF(VP, FN, FP, VN)#Calculo del valor Fpara las Consonantes
        fprf = falsePositive(VP, FN, FP, VN)#Calculo del valor FPR para las Consonantes
    
        lista_sensibilidad.append(sensibilidadf) #Se rellena la lista de las sensibilidades 
        lista_Especificidad.append(especificidadf) #Se rellena la lista de la especificidad  
        lista_Exactitud.append(exactitudf) #Se rellena la lista de la exactitud
        lista_Precision.append(precisionf) #Se rellena la lista de la precision 
        lista_Matthews.append(matthewf) #Se rellena la lista del coeficiente de Matthew 
        lista_Jaccard.append(jaccardf) #Se rellena la lista del Indice de Jaccard 
        lista_ParametroD.append(parametroDf) #Se rellena la lista del Parámetro d'
        lista_ErrorMedio.append(errorMedf) #Se rellena la lista del error medio 
        lista_ValorF.append(valorFf) #Se rellenas la lista del valor F 
        lista_FPR.append(fprf) #Se rellena la lista del FPR 
        lista_Letra_Posicion.append(Nombre_Letra) #Nombre de las letras que vamos iterando
    
    #Se transforma el indice de la matriz/dataframe numerico por un incide que refleja la letra estudiada
    Indices = lista_Letra_Posicion
    
    #Se crea un dataframe con los resultados obtenidos para todos los parametros estadisticos calculados
    Resultado_Estadistico = pd.DataFrame({"Sensibilidad": lista_sensibilidad, "Especificidad": lista_Especificidad, 
                                          "Exactitud": lista_Exactitud, "Precision": lista_Precision,
                                          "Error_Medio": lista_ErrorMedio, "Valor_F": lista_ValorF ,
                                          "Ratio_FP": lista_FPR, "Coeficiente_Matthews": lista_Matthews,
                                          "Indice_Jaccard": lista_Jaccard, "Parametro_d": lista_ParametroD}, index = Indices)
    
    #Se imprime la tabla (dataframe) con los resultados totales                                       
    print("\n Los resultados estadisticos son: \n \n", Resultado_Estadistico,"\n") 
    
    #La funcion devuelve el dataframe "Resultado_Estadistico", con los parametros estadisticos calculados
    return Resultado_Estadistico




##### PARAMETROS ESTADISTICOS INDIVIDUALES #####
    
#Sensibilidad
def EstatSensibilidad(matriz):
    sensib = pd.DataFrame(data = FuncionesEstadisticas(matriz).Sensibilidad);
    print("La Sensibilidad es: \n \n", sensib)
    return sensib

#Especificidad
def EstatEspecificidad(matriz):
    espec = pd.DataFrame(data = FuncionesEstadisticas(matriz).Especificidad);
    print("La Especificidad es: \n \n", espec)
    return espec

#Exactitud
def EstatExactitud(matriz):
    exact = pd.DataFrame(data = FuncionesEstadisticas(matriz).Exactitud);
    print("La Exactitud es: \n \n", exact)
    return exact

#Precision
def EstatPrecision(matriz):
    precis= pd.DataFrame(data = FuncionesEstadisticas(matriz).Precision);
    print("La Precision es: \n \n", precis)
    return precis

#EError Medio
def EstatError_Medio(matriz):
    errorm = pd.DataFrame(data = FuncionesEstadisticas(matriz).Error_Medio);
    print("El Error Medio es: \n \n", errorm)
    return errorm

#Valor F
def EstatValor_F(matriz):
    vf = pd.DataFrame(data = FuncionesEstadisticas(matriz).Valor_F);
    print("El Valor-F es: \n \n", vf)
    return vf

#Ratio de Falsos Positivos
def EstatRatio_FP(matriz):
    ratFP = pd.DataFrame(data = FuncionesEstadisticas(matriz).Ratio_FP);
    print("El Ratio de Falsos Positivos es: \n \n", ratFP)
    return ratFP

#Coeficiente de Matthews
def EstatMatthews(matriz):
    matt = pd.DataFrame(data = FuncionesEstadisticas(matriz).Coeficiente_Matthews);
    print("El Coeficiente de Matthews es: \n \n", matt)
    return matt

#Indice de Jaccard
def EstatJaccard(matriz):
    jacc = pd.DataFrame(data = FuncionesEstadisticas(matriz).Indice_Jaccard);
    print("El Indice de Jaccard es: \n \n", jacc)
    return jacc

#Parametro D'
def EstatParamD(matriz):
    parD = pd.DataFrame(data = FuncionesEstadisticas(matriz).Parametro_d);
    print("El Parametro D es: \n \n", parD)
    return parD





##### GUARDAR LOS DATAFRAMES RESULTANTES #####
    
#Funcion para guardar los dataframes obtenidos en formato .xlsx
def dfToExcel(dataframe, nombre):
    dataframe.to_excel(nombre) 
    

