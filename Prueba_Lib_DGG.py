#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jun  8 14:44:05 2020

@author: davidgomezgomez
"""

import Lib_RAH_DGG as ld
import os
import pandas as pd


#En primer lugar, se ipmorta el archivo de texto y se transforma a un dataframe

#Sirve para establecer en qué carpeta se está trabajando (ruta)
working_path = os.getcwd() 

#Se construye un dataframe importando el archivo txt separado por tabuladores
texto_kaldi = pd.read_csv("salida.txt",delimiter="\t") 

#Se cambia (acorta) el de la columna que indica el nº de sílabas para poder trabajar con ella más facilmente
texto_kaldi.rename(columns={'Number of Syllables': 'NSyllables'}, inplace = True) 





##### 1. PORCENTAJE ACIERTO #####
#Muestra en pantalla del porcentaje de Aciertos para palabras con 2, 3 y 4 silabas. Funcion: PorcentajeAciertos
ld.PorcentajeAciertos(texto_kaldi)





##### 2. MATRICES DE CONFUSION #####
# 2.1 VOCALES
#Calculo de matriz de confusion Vocales. Funcion: ConfusionMatrixVoc(matriz)
Matriz_Confusion_V = pd.DataFrame.from_dict(ld.ConfusionMatrixVoc(texto_kaldi))
#Calculo de matriz de confusion Normalizada Vocales. Funcion: NormalConfusionMatrixVoc
Normal_Matriz_Confusion_V = pd.DataFrame.from_dict(ld.NormalConfusionMatrixVoc(texto_kaldi))

# 2.2 CONSONANTES
#Calculo de matriz de confusion Consonantes. Funcion: ConfusionMatrixCons(matriz)
Matriz_Confusion_C = pd.DataFrame.from_dict(ld.ConfusionMatrixCons(texto_kaldi))
#Calculo de matriz de confusion Consonantes. Funcion: NormalConfusionMatrixCons
Normal_Matriz_Confusion_C = pd.DataFrame.from_dict(ld.NormalConfusionMatrixCons(texto_kaldi))

# 2.2.1 SONORIDAD: Sordas o Sonoras
Sonoridad_Sonora = pd.DataFrame.from_dict(ld.CSonora(texto_kaldi))
Sonoridad_Sorda = pd.DataFrame.from_dict(ld.CSorda(texto_kaldi))
#Normalizadas
Normal_Sonoridad_Sonora = pd.DataFrame.from_dict(ld.NormalizeCSonora(texto_kaldi))
Normal_Sonoridad_Sorda = pd.DataFrame.from_dict(ld.NormalizeCSorda(texto_kaldi))

# 2.2.2 LUGAR DE ARTICULACION: Frontal, Coronal, Back
Lugar_Frontal = pd.DataFrame.from_dict(ld.CFrontal(texto_kaldi))
Lugar_Coronal = pd.DataFrame.from_dict(ld.CCoronal(texto_kaldi))
Lugar_Back = pd.DataFrame.from_dict(ld.CBack(texto_kaldi))
#Normalizadas
Normal_Lugar_Frontal = pd.DataFrame.from_dict(ld.NormalizeCFrontal(texto_kaldi))
Normal_Lugar_Coronal = pd.DataFrame.from_dict(ld.NormalizeCCoronal(texto_kaldi))
Normal_Lugar_Back = pd.DataFrame.from_dict(ld.NormalizeCBack(texto_kaldi))

# 2.2.3 MODO DE ARTICULACION: Oclusiva, Africada, Fricativa, Nasal, Aproximantes
Modo_Oclusiva = pd.DataFrame.from_dict(ld.COclusiva(texto_kaldi))
Modo_Africada = pd.DataFrame.from_dict(ld.CAfricada(texto_kaldi))
Modo_Fricativa = pd.DataFrame.from_dict(ld.CFricativa(texto_kaldi))
Modo_Nasal = pd.DataFrame.from_dict(ld.CNasal(texto_kaldi))
Modo_Aproximante = pd.DataFrame.from_dict(ld.CAproximante(texto_kaldi))

#Normalizadas
Normal_Modo_Oclusiva = pd.DataFrame.from_dict(ld.NormalizeCOclusiva(texto_kaldi))
Normal_Modo_Africada = pd.DataFrame.from_dict(ld.NormalizeCAfricada(texto_kaldi))
Normal_Modo_Fricativa = pd.DataFrame.from_dict(ld.NormalizeCFricativa(texto_kaldi))
Normal_Modo_Nasal = pd.DataFrame.from_dict(ld.NormalizeCNasal(texto_kaldi))
Normal_Modo_Aproximante = pd.DataFrame.from_dict(ld.NormalizeCAproximante(texto_kaldi))





#### 3. IMPRIMIR MATTRICES ####

# Visualizacion de las matrices normalizadas para ver con claridad fallos. 

## 3.1 PLOT NARANJA-AZUL
# Color = Escala Naranja-Azul Funcion: PlotMatOrangeBlue(matriz, "nombre")

#3.1.1 Visualizacion matriz de consonantes y vocales
ld.PlotMatOrangeBlue(Normal_Matriz_Confusion_V, "Vocales")
ld.PlotMatOrangeBlue(Normal_Matriz_Confusion_C, "Consonantes")

# 3.1.2 Visualizacion matrices segun Sonoridad
ld.PlotMatOrangeBlue(Normal_Sonoridad_Sonora, "Sonoridad Sonora")
ld.PlotMatOrangeBlue(Normal_Sonoridad_Sorda, "Sonoridad Sorda")

# 3.1.3 Visualizacion matrices segun el Lugar de articulacion
ld.PlotMatOrangeBlue(Normal_Lugar_Frontal, "Lugar Frontal")
ld.PlotMatOrangeBlue(Normal_Lugar_Coronal, "Lugar Coronal")
ld.PlotMatOrangeBlue(Normal_Lugar_Back, "Lugar Back")

# 3.1.4 Visualizacion matrices segun el Modo de articulacion
ld.PlotMatOrangeBlue(Normal_Modo_Oclusiva, "Modo Oclusiva")
ld.PlotMatOrangeBlue(Normal_Modo_Africada, "Modo Africada")
ld.PlotMatOrangeBlue(Normal_Modo_Fricativa, "Modo Fricativa")
ld.PlotMatOrangeBlue(Normal_Modo_Nasal, "Modo Nasal")
ld.PlotMatOrangeBlue(Normal_Modo_Aproximante, "Modo Aproximante")


## 3.2 PLOT VIRIDIS
# Color = Escala Vuridis Funcion: PlotMatViridis(matriz, "nombre")

#3.2.1 Visualizacion matriz de consonantes y vocales
ld.PlotMatViridis(Normal_Matriz_Confusion_V, "Vocales")
ld.PlotMatViridis(Normal_Matriz_Confusion_C, "Consonantes")

# 3.2.2 Visualizacion matrices segun Sonoridad
ld.PlotMatViridis(Normal_Sonoridad_Sonora, "Sonoridad Sonora")
ld.PlotMatViridis(Normal_Sonoridad_Sorda, "Sonoridad Sorda")

# 3.2.3 Visualizacion matrices segun el Lugar de articulacion
ld.PlotMatViridis(Normal_Lugar_Frontal, "Lugar Frontal")
ld.PlotMatViridis(Normal_Lugar_Coronal, "Lugar Coronal")
ld.PlotMatViridis(Normal_Lugar_Back, "Lugar Back")

# 3.2.4 Visualizacion matrices segun el Modo de articulacion
ld.PlotMatViridis(Normal_Modo_Oclusiva, "Modo Oclusiva")
ld.PlotMatViridis(Normal_Modo_Africada, "Modo Africada")
ld.PlotMatViridis(Normal_Modo_Fricativa, "Modo Fricativa")
ld.PlotMatViridis(Normal_Modo_Nasal, "Modo Nasal")
ld.PlotMatViridis(Normal_Modo_Aproximante, "Modo Aproximante")





#### 4. LAS MATRICES 2X2 OBTENIDAS #####
#Obtener e (imprimir en pantalla) las matrices 2x2 para cada letra; se crea una lista en la que cada elemento es la matriz 2x2 de una letra 
#Funcion: Matriz2x2(matriz)

# Matriz 2x2 Para vocales y consonantes
Matrices2x2_Vocales = ld.Matriz2x2(Matriz_Confusion_V)
Matrices2x2_Consonantes = ld.Matriz2x2(Matriz_Confusion_C)
#print(Matrices2x2_Vocales)
#print(Matrices2x2_Consonantes)

# Matriz 2x2 Segun Sonoridad
Matrices2x2_Sonora = ld.Matriz2x2(Sonoridad_Sonora)
Matrices2x2_Sorda = ld.Matriz2x2(Sonoridad_Sorda)
#print(Matrices2x2_Sonora)
#print(Matrices2x2_Sorda)

# Matriz 2x2 Segun Lugar de Ariculacion
Matrices2x2_Frontal = ld.Matriz2x2(Lugar_Frontal)
Matrices2x2_Coronal = ld.Matriz2x2(Lugar_Coronal)
Matrices2x2_Back = ld.Matriz2x2(Lugar_Back)
#print(Matrices2x2_Frontal)
#print(Matrices2x2_Coronal)
#print(Matrices2x2_Back)

# Matriz 2x2 Segun Modo de Ariculacion
Matrices2x2_Oclusiva = ld.Matriz2x2(Modo_Oclusiva)
Matrices2x2_Africada = ld.Matriz2x2(Modo_Africada)
Matrices2x2_Fricativa = ld.Matriz2x2(Modo_Fricativa)
Matrices2x2_Nasal = ld.Matriz2x2(Modo_Nasal)
Matrices2x2_Aproximante = ld.Matriz2x2(Modo_Aproximante)
#print(Matrices2x2_Oclusiva)
#print(Matrices2x2_Africada)
#print(Matrices2x2_Fricativa)
#print(Matrices2x2_Nasal)
#print(Matrices2x2_Aproximante)





##### 5. OBTENCION DE DATOS ESTADISTICOS GENERALES#####
# Obtencion e impresion en pantalla los datos estadisticos y creacion de un dataframe. Funcion: FuncionesEstadisticas(matriz)

#5.1 Datos estadisticos para vocales y consonantes
Estadistica_Vocales = pd.DataFrame.from_dict(ld.FuncionesEstadisticas(Matriz_Confusion_V))
Estadistica_Consonantes = pd.DataFrame.from_dict(ld.FuncionesEstadisticas(Matriz_Confusion_C))
print("\n Los parametros estadisticos obtenidos para las vocales son: \n ", Estadistica_Vocales)
print("\n Los parametros estadisticos obtenidos para las consonantes son: \n ", Estadistica_Consonantes)

# 5.2 Datos estadisticos sonoridad
Estadistica_Sonora = pd.DataFrame.from_dict(ld.FuncionesEstadisticas(Sonoridad_Sonora))
Estadistica_Sorda = pd.DataFrame.from_dict(ld.FuncionesEstadisticas(Sonoridad_Sorda))
print("\n Los parametros estadisticos obtenidos para las Consonantes Sonoras son: \n ", Estadistica_Sonora)
print("\n Los parametros estadisticos obtenidos para las Consonantes Sordas son: \n ", Estadistica_Sorda)

# 5.3 Datos estadisticos lugar de articulacion
Estadistica_Frontal = pd.DataFrame.from_dict(ld.FuncionesEstadisticas(Lugar_Frontal))
Estadistica_Coronal = pd.DataFrame.from_dict(ld.FuncionesEstadisticas(Lugar_Coronal))
Estadistica_Back = pd.DataFrame.from_dict(ld.FuncionesEstadisticas(Lugar_Back))
print("\n Los parametros estadisticos obtenidos para las Consonantes Frontales son: \n ", Estadistica_Frontal)
print("\n Los parametros estadisticos obtenidos para las Consonantes Coronales son: \n ", Estadistica_Coronal)
print("\n Los parametros estadisticos obtenidos para las Consonantes en Back son: \n ", Estadistica_Back)

# 5.4 Datos estadisticos lugar de articulacion
Estadistica_Oclusiva = pd.DataFrame.from_dict(ld.FuncionesEstadisticas(Modo_Oclusiva))
#Estadistica_Africada = pd.DataFrame.from_dict(ld.FuncionesEstadisticas(Modo_Africada)) #No hay estadisticas, solo es un dato
Estadistica_Fricativa = pd.DataFrame.from_dict(ld.FuncionesEstadisticas(Modo_Fricativa))
Estadistica_Nasal = pd.DataFrame.from_dict(ld.FuncionesEstadisticas(Modo_Nasal))
Estadistica_Aproximante = pd.DataFrame.from_dict(ld.FuncionesEstadisticas(Modo_Aproximante))
print("\n Los parametros estadisticos obtenidos para las Consonantes Oclusivas son: \n ", Estadistica_Oclusiva)
#print("\n Los parametros estadisticos obtenidos para las Consonantes Africadas son: \n ", Estadistica_Africada) #No hay estadisticas, solo es un dato
print("\n Los parametros estadisticos obtenidos para las Consonantes Fricativas son: \n ", Estadistica_Fricativa)
print("\n Los parametros estadisticos obtenidos para las Consonantes Nasales son: \n ", Estadistica_Nasal)
print("\n Los parametros estadisticos obtenidos para las Consonantes Aproximantes son: \n ", Estadistica_Aproximante)





##### 6. OBTENCION DE PARAMETROS ESTADISTICOS CONCRETOS #####
#Se realiza la prueba obteniendo los parametros estadisticos para las Consonantes

Sensibilidad_Consonantes = ld.EstatSensibilidad(Matriz_Confusion_C)

Especificidad_Consonantes = ld.EstatEspecificidad(Matriz_Confusion_C)

Exactitud_Consonantes = ld.EstatExactitud(Matriz_Confusion_C)

Precision_Consonantes  = ld.EstatPrecision(Matriz_Confusion_C)

Error_Medio_Consonantes = ld.EstatError_Medio(Matriz_Confusion_C)

Valor_F_Consonantes = ld.EstatValor_F(Matriz_Confusion_C)

Ratio_FP_Consonantes = ld.EstatRatio_FP(Matriz_Confusion_C)

Coeficiente_Matthews_Consonantes = ld.EstatMatthews(Matriz_Confusion_C)

Indice_Jaccard_Consonantes = ld.EstatJaccard(Matriz_Confusion_C)

Parametro_D_Consonantes = ld.EstatParamD(Matriz_Confusion_C)


 
    

##### 7. GUARDAR LOS RESULTADOS DE DATAFRAME #####

#Se guarda el dataframe de las Consonantes en formato .xlsx
ld.dfToExcel(Normal_Matriz_Confusion_C, "Consonantes_Normalizadas.xlsx")

