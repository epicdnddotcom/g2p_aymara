"""
    Programa para cargar y utilizar un modelo predictor de fonemas
"""
# importamos la clase
from Graph2PhonModel import Graph2PhonModel

# directorios necesarios para cargar el dataset y los modelos
dic_dir = "dic_datasets/aymara3.dic"
model_dir ="aymaramodel4/"
model_name = "aymara4"

# crea el modelo y carga los conjuntos de entrenamiento, validacion y testeo
g2pModel = Graph2PhonModel(dic_dir, model_dir=model_dir, name=model_name)
# carga un modelo entrenado almacenado en los directorios definidos y con el formato adecuado
trained = g2pModel.loadModel()
#realiza predicciones sobre el conjunto de testeo
g2pModel.testModel(trained)
# con el modelo entrenado podemos realizar predicciones de nuevas palabras
#g2pModel.predictPhoneme("janiwa", trained)
# existe tambien un modo interactivo
#g2pModel.runInteractive(trained)
