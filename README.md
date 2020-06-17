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


