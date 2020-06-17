# Lib_RAH_DGG
Desarrollo de una librería Python para la evaluación de los resultados obtenidos con Kaldi en problemas de Reconocimiento Automático del Habla (RAH).

**0.	Introducción**

Esta librería de funciones permite obtener los parámetros estadísticos con los que determinar la exactitud y corregir errores de un sistema de RAH (Reconocimiento Automático del Habla) en el software Kaldi (https://kaldi-asr.org/). Para ello, es necesario contar con los resultados/datos devueltos por el sistema en un formato numérico.

Esta librería surge por la necesidad de evaluar la precisión y posibles errores de un entrenamiento y receta aplicada en el software Kaldi de una manera rápida, obteniendo unos resultados simples de identificar.

**1.	Ejemplos de los resultados obtenidos**

Para realizar estos ejemplos, se ha utilizado la receta ASICA de Kaldi, creada por D. Ignacio Moreno-Torres Sánchez y cuyo enlace a GitHu es:

( https://github.com/Caliope-SpeechProcessingLab/ASICAKaldiRecipe_old )

Al instalar la receta ASICA en el software Kaldi y ejecutar el script “result_format.py”, se obtiene un archivo de texto llamado “salida.txt”, el cuál contiene múltiples parámetros relacionados con los resultados obtenidos para la prueba de RAH realizada; estos parámetros en el archivo de texto están separados por tabuladores. Para los ejemplos realizados y comprobación del correcto funcionamiento de la librería, se utilizará el archivo “salida.txt” contenido en este repositorio.

Algunos ejemplos de resultados obtenidos con la librería “Lib_RAH_DGG.py” son:

- Obtención de un mapa de una figura para una matriz de confusión de consonantes:
  ```
  Lib_RAH_DGG.PlotMatViridis(Normal_Matriz_Confusion_C, "Consonantes")
  ```
  Se obtiene como resultado la siguiente figura:
  
  ![alt text](Imagenes_Ejemplo_Resultados_Librería/Matriz_Viridis_Consonantes_salidatxt.png)

- Obtención de una tabla de valores estadísticos para cada una de las consonantes reconocidas por el sistema de RAH:
  ```
  Lib_RAH_DGG.FuncionesEstadisticas(Matriz_Confusion_C)
  ```
  Se obtiene como resultado la siguiente figura:
  
  ![alt text](Imagenes_Ejemplo_Resultados_Librería/Funciones_Estadisticas_Consonantes_salidatxt.png)
  
  ***2.	Configuración y uso***
  
- Prerrequisitos
  
  Para utilizar esta librería, es necesario tener instalado Python 3.
  
- Descarga

  Puedes descargar esta librería desde el siguiente enlace:
  
  (https://github.com/davidgg97/Python_Statistics_Library_for_Kaldi_RAH_DGG/blob/master/Lib_RAH_DGG.py)
  
- Uso de las diferentes funciones

  Las funciones disponibles con esta librería son:

  -	Función para el cálculo del porcentaje de acierto (sílabas reconocidas) en el RAH. Tiene como parámetro de entrada un     dataframe con los datos que se quieren analizar:
  
  ```
  PorcentajeAciertos(dataframe)
  ```
  
  - Funciones de obtención de matrices de confusión para vocales, consonantes y consonantes clasificadas según su sonoridad, lugar de articulación y modo de articulación. Tienen como parámetro de entrada el dataframe con los datos que se quieren analizar:
  
  ```
  ConfusionMatrixVoc(dataframe_datos)
  ConfusionMatrixCons(dataframe_datos)
  CSonora(dataframe_datos)
  CSorda(dataframe_datos)
  CFrontal(dataframe_datos)
  CCoronal(dataframe_datos)
  CBack(dataframe_datos)
  COclusiva(dataframe_datos)
  CAfricada(dataframe_datos)
  CFricativa(dataframe_datos)
  CNasal(dataframe_datos)
  CAproximante(dataframe_datos)
  ```
  
  - Funciones para calcular las matrices anteriores normalizadas. Tienen como parámetro de entrada el dataframe con los datos que se quieren analizar:
  
  ```
  NormalConfusionMatrixVoc(dataframe_datos)
  NormalConfusionMatrixCons(dataframe_datos)
  NormalizeCSonora(dataframe_datos)
  NormalizeCSorda(dataframe_datos)
  NormalizeCFrontal(dataframe_datos)
  NormalizeCCoronal(dataframe_datos)
  NormalizeCBack(dataframe_datos)
  NormalizeCOclusiva(dataframe_datos)
  NormalizeCAfricada(dataframe_datos)
  NormalizeCFricativa(dataframe_datos)
  NormalizeCNasal(dataframe_datos)
  NormalizeCAproximante(dataframe_datos)
  ```
  
  - Funciones para graficar (obtiene una figura) de las matrices de confusión normalizadas para las letras que se quiere reconocer. La matriz graficada puede tener una gradación de color naranja-azul o viridis. Tienen como parámetros de entrada el dataframe con la matriz a graficar y el nombre que se le quiere dar a la figura (debe escribirse entrecomillado):
  
  ```
  PlotMat OrangeBlue(dataframe_matriz_normalizada, "Nombre")
  PlotMatViridis(dataframe_matriz_normalizada, "Nombre")
  ```
  
  - Función para transformar matrices de confusión MxM de varios parámetros, en submatrices de un solo parámetro 2x2. Tiene como parámetro de entrada el dataframe de la matriz que se quiera transformar.
  
  ```
  Matriz2x2(dataframe_matriz)
  ```
  
  - Funciones estadísticas, se podrá obtener una función estadística concreta para una letra concreta (conociéndose sus variables de Verdaderos Positivos, Falsos Positivos, Falsos Negativos, Verdaderos Negativos). Tienen como parámetros de entrada estas variables. Las funciones estadísticas calculan la Sensibilidad, Especificidad, Exactitud, Precisión, Error Medio, Valor-F, Ratio de Falsos Positivos, Coeficiente de Matthews, Índice de Jaccar y Parámetro D’.
  
  ```
  sensibilidad(VP, FN, FP, VN)
  especificidad(VP, FN, FP, VN)
  exactitud(VP, FN, FP, VN)
  precision(VP, FN, FP, VN)
  errorMedio (VP, FN, FP, VN)
  valorF(VP, FN, FP, VN)
  falsePositive(VP, FN, FP, VN)
  matthew(VP,FN, FP, VN)
  jaccard(VP, FN, FP, VN)
  parametroD(VP, FN, FP, VN)
  ```
  
  - Funciones para calcular los parámetros estadísticos anteriores, para todas las letras de una matriz de confusión. Tienen como parámetro de entrada la matriz de confusión que se quiera estudiar.
  
  ```
  EstatSensibilidad(matriz_confusion)
  EstatEspecificidad(matriz_confusion)
  EstatExactitud(matriz_confusion)
  EstatPrecision(matriz_confusion)
  EstatError_Medio(matriz_confusion)
  EstatValor_F(matriz_confusion)
  EstatRatio_FP(matriz_confusion)
  EstatMatthews(matriz_confusion)
  EstatJaccard(matriz_confusion)
  EstatParamD(matriz_confusion)
  ```

  - Función para calcular los parámetros estadísticos anteriores y, obtener una salida conjunta en una única tabla. Tiene como parámetro de entrada la matriz de confusión de las letras que se quiera estudiar.

  ```
   FuncionesEstadisticas(matriz_confusion)
  ```

  - Función para guardar los resultados de un dataframe en un archivo “.xlsx”:

  ```
  dfToExcel(Normal_Matriz_Confusion,"Nombre_archivo.xlsx")
  ```

***3.	Información***

Esta librería está en fase de desarrollo, ha sido creada por D. David Gómez Gómez para el desarrollo de su Trabajo de Fin de Grado (Desarrollo de una librería Python para la evaluación de los resultados obtenidos con Kaldi en problemas de Reconocimiento Automático del Habla (RAH)), el cuál ha sido tutorizado por los profesores D. Enrique Nava Baro y D. Ignacio Moreno-Torres Sánchez.



