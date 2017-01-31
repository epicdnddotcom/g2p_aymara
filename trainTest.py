"""
    Programa para entrenar un modelo predictor de fonemas
"""
# importamos la clase
from Graph2PhonModel import Graph2PhonModel
# directorios necesarios para guardar y cargar el dataset y los modelos
dic_dir = "dic_datasets/aymara2.dic"
model_dir ="noeli/"
model_name = "noeli"
# crea el modelo y carga los conjuntos de entrenamiento, validacion y testeo
g2pModel = Graph2PhonModel(dic_dir, model_dir=model_dir, name=model_name)
# crea la arquitectura del modelo de la red neuronal
g2pModel.prepareModel()
# almacena la arquitectura y los hiperparametros del modelo en un archivo .json
g2pModel.saveModel()
# entrena el modelo con las iteraciones definidas por el usuario
#se evalua la precision con el conjunto de validacion en cada iteracion
#y se almacenan los pesos correspondientes al mejor rendimiento.
#Al mismo tiempo se almacenan indicadores de rendimiento para futuros analisis
trained = g2pModel.trainModel()
#realiza predicciones sobre el conjunto de testeo
g2pModel.testModel(trained)
# con el modelo entrenado podemos realizar predicciones de nuevas palabras
g2pModel.predictPhoneme("janiwa", trained)
# existe tambien un modo interactivo
g2pModel.runInteractive(trained)
