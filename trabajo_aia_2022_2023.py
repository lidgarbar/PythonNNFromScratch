#!/usr/bin/env python3
# -*- coding: utf-8 -*-
 
# ===================================================================
# Ampliación de Inteligencia Artificial, 2022-23
# PARTE I del trabajo práctico: Implementación de regresión logística
# Dpto. de CC. de la Computación e I.A. (Univ. de Sevilla)
# ===================================================================


# --------------------------------------------------------------------------
# Autor(a) del trabajo:
#
# APELLIDOS:
# NOMBRE: 
#
# Segundo(a) componente (si se trata de un grupo):
#
# APELLIDOS:
# NOMBRE:
# ----------------------------------------------------------------------------


# ****************************************************************************************
# HONESTIDAD ACADÉMICA Y COPIAS: un trabajo práctico es un examen. La discusión 
# y el intercambio de información de carácter general con los compañeros se permite, 
# pero NO AL NIVEL DE CÓDIGO. Igualmente el remitir código de terceros, OBTENIDO A TRAVÉS
# DE LA RED o cualquier otro medio, se considerará plagio. En particular no se 
# permiten implementaciones obtenidas con HERRAMIENTAS DE GENERACIÓN AUTOMÁTICA DE CÓDIGO. 
# Si tienen dificultades para realizar el ejercicio, consulten con el profesor. 
# En caso de detectarse plagio (previamente con aplicaciones anti-plagio o durante 
# la defensa, si no se demuestra la autoría mediante explicaciones convincentes), 
# supondrá una CALIFICACIÓN DE CERO en la asignatura, para todos los alumnos involucrados. 
# Sin perjuicio de las medidas disciplinarias que se pudieran tomar. 
# *****************************************************************************************


# IMPORTANTE: NO CAMBIAR EL NOMBRE NI A ESTE ARCHIVO NI A LAS CLASES, MÉTODOS
# Y ATRIBUTOS QUE SE PIDEN. EN PARTICULAR: NO HACERLO EN UN NOTEBOOK.

# NOTAS: 
# * En este trabajo NO SE PERMITE usar Scikit Learn (excepto las funciones que
#   se usan en carga_datos.py). 

# * SE RECOMIENDA y SE VALORA especialmente usar numpy. Las implementaciones 
#   saldrán mucho más cortas y eficientes, y se puntuarÁn mejor.   

import numpy as np

# *****************************************
# CONJUNTOS DE DATOS A USAR EN ESTE TRABAJO
# *****************************************

# Para aplicar las implementaciones que se piden en este trabajo, vamos a usar
# los siguientes conjuntos de datos. Para cargar todos los conjuntos de datos,
# basta con descomprimir el archivo datos-trabajo-aia.tgz y ejecutar el
# archivo carga_datos.py (algunos de estos conjuntos de datos se cargan usando
# utilidades de Scikit Learn, por lo que para que la carga se haga sin
# problemas, deberá estar instalado el módulo sklearn). Todos los datos se
# cargan en arrays de numpy:

# * Datos sobre concesión de prestamos en una entidad bancaria. En el propio
#   archivo datos/credito.py se describe con más detalle. Se carga en las
#   variables X_credito, y_credito.   

# * Conjunto de datos de la planta del iris. Se carga en las variables X_iris,
#   y_iris.  

# * Datos sobre votos de cada uno de los 435 congresitas de Estados Unidos en
#   17 votaciones realizadas durante 1984. Se trata de clasificar el partido al
#   que pertenece un congresita (republicano o demócrata) en función de lo
#   votado durante ese año. Se carga en las variables X_votos, y_votos. 

# * Datos de la Universidad de Wisconsin sobre posible imágenes de cáncer de
#   mama, en función de una serie de características calculadas a partir de la
#   imagen del tumor. Se carga en las variables X_cancer, y_cancer.
  
# * Críticas de cine en IMDB, clasificadas como positivas o negativas. El
#   conjunto de datos que usaremos es sólo una parte de los textos. Los textos
#   se han vectorizado usando CountVectorizer de Scikit Learn, con la opción
#   binary=True. Como vocabulario, se han usado las 609 palabras que ocurren
#   más frecuentemente en las distintas críticas. La vectorización binaria
#   convierte cada texto en un vector de 0s y 1s en la que cada componente indica
#   si el correspondiente término del vocabulario ocurre (1) o no ocurre (0)
#   en el texto (ver detalles en el archivo carga_datos.py). Los datos se
#   cargan finalmente en las variables X_train_imdb, X_test_imdb, y_train_imdb,
#   y_test_imdb.    

# * Un conjunto de imágenes (en formato texto), con una gran cantidad de
#   dígitos (de 0 a 9) escritos a mano por diferentes personas, tomado de la
#   base de datos MNIST. En digitdata.zip están todos los datos en formato
#   comprimido. Para preparar estos datos habrá que escribir funciones que los
#   extraigan de los ficheros de texto (más adelante se dan más detalles). 



# ==================================================
# EJERCICIO 1: SEPARACIÓN EN ENTRENAMIENTO Y PRUEBA 
# ==================================================

# Definir una función 
def particion_entr_prueba(X,y,test=0.20):
    X_train=[];X_test=[];y_train=[];y_test=[]
    #Vemos cuantas clases tiene nuestros datos de entrada 
    clases = np.unique(y);
        
    #Inicializamos una lista que contendrá los indices de los datos destinados al test
    indices=[]
    #Por cada clase de nuestros datos
    for clase in clases:
        #Nos quedamos con todos los indices del conjunto que sean clasidicados con esa clase 
        valores=[i for i in range(len(X)) if y[i]==clase]
       
        #Calculamos cuantos datos de dicha clase va a tener el test
        num=int(len(valores)*test)
                
        #Hacemos un random para que sea aleatorio
        np.random.shuffle(valores)
        
        #Añadimos a nuestra lista de indices para el test el numero de indices calculado antes  
        indices.extend(valores[:num])
        
    #Ordenamos los indices para que haya consistencia como nos pide en el enunciado
    indices.sort()

    #Calculamos los índices para train que serán aquellos que no hemos cogido en test
    train_ind=[i for i in range(len(X)) if i not in indices]
  
    #Cargamos los datos correspondientes a cada conjunto
    X_test.extend(X[indices])
    y_test.extend(y[indices])    
    X_train.extend(X[train_ind])
    y_train.extend(y[train_ind])        
    
    #Lo transformamos en un array para devolverlo
    X_train = np.array(X_train)
    X_test = np.array(X_test)
    y_train = np.array(y_train)
    y_test = np.array(y_test)
    
    #Si estamos antes una clasificacion binaria y las clses no son 0s o 1s
    if(len(clases)==2 and (clases is not [0,1] or clases is not [1,0])):
        #Cambiamos los valores por 0 cuando sean de la primera clase y 1 en otro caso
        y_train=np.where(y_train==clases[0],0,1)
        y_test=np.where(y_test==clases[0],0,1)
        
    return X_train, X_test, y_train, y_test  

# que recibiendo un conjunto de datos X, y sus correspondientes valores de
# clasificación y, divide ambos en datos de entrenamiento y prueba, en la
# proporción marcada por el argumento test. La división ha de ser ALEATORIA y
# ESTRATIFICADA respecto del valor de clasificación. Por supuesto, en el orden 
# en el que los datos y los valores de clasificación respectivos aparecen en
# cada partición debe ser consistente con el orden original en X e y.   

# ------------------------------------------------------------------------------
# Ejemplos:
# =========

# En votos:
# >>> Xe_votos,Xp_votos,ye_votos,yp_votos=particion_entr_prueba(X_votos,y_votos,test=1/3)

# Como se observa, se han separado 2/3 para entrenamiento y 1/3 para prueba:
# >>> y_votos.shape[0],ye_votos.shape[0],yp_votos.shape[0]
#    (435, 290, 145)

# Las proporciones entre las clases son (aprox) las mismas en los dos conjuntos de
# datos, y la misma que en el total: 267/168=178/112=89/56

# >>> np.unique(y_votos,return_counts=True)
#  (array([0, 1]), array([168, 267]))
# >>> np.unique(ye_votos,return_counts=True)
#  (array([0, 1]), array([112, 178]))
# >>> np.unique(yp_votos,return_counts=True)
#  (array([0, 1]), array([56, 89]))

# La división en trozos es aleatoria y, por supuesto, en el orden en el que
# aparecen los datos en Xe_votos,ye_votos y en Xp_votos,yp_votos, se preserva
# la correspondencia original que hay en X_votos,y_votos.


# Otro ejemplo con los datos del cáncer, en el que se observa que las proporciones
# entre clases se conservan en la partición. 
    
# >>> Xev_cancer,Xp_cancer,yev_cancer,yp_cancer=particion_entr_prueba(X_cancer,y_cancer,test=0.2)

# >>> np.unique(y_cancer,return_counts=True)
# (array([0, 1]), array([212, 357]))

# >>> np.unique(yev_cancer,return_counts=True)
# (array([0, 1]), array([170, 286]))

# >>> np.unique(yp_cancer,return_counts=True)
# (array([0, 1]), array([42, 71]))    


# Podemos ahora separar Xev_cancer, yev_cancer, en datos para entrenamiento y en 
# datos para validación.

# >>> Xe_cancer,Xv_cancer,ye_cancer,yv_cancer=particion_entr_prueba(Xev_cancer,yev_cancer,test=0.2)

# >>> np.unique(ye_cancer,return_counts=True)
# (array([0, 1]), array([170, 286]))

# >>> np.unique(yv_cancer,return_counts=True)
# (array([0, 1]), array([170, 286]))


# Otro ejemplo con más de dos clases:

# >>> Xe_credito,Xp_credito,ye_credito,yp_credito=particion_entr_prueba(X_credito,y_credito,test=0.4)

# >>> np.unique(y_credito,return_counts=True)
# (array(['conceder', 'estudiar', 'no conceder'], dtype='<U11'),
#  array([202, 228, 220]))

# >>> np.unique(ye_credito,return_counts=True)
# (array(['conceder', 'estudiar', 'no conceder'], dtype='<U11'),
#  array([121, 137, 132]))

# >>> np.unique(yp_credito,return_counts=True)
# (array(['conceder', 'estudiar', 'no conceder'], dtype='<U11'),
#  array([81, 91, 88]))
# ------------------------------------------------------------------


## ---------- 



















# ===========================
# EJERCICIO 2: NORMALIZADORES
# ===========================

# En esta sección vamos a definir dos maneras de normalizar los datos. De manera 
# similar a como está diseñado en scikit-learn, definiremos un normalizador mediante
# una clase con un metodo "ajusta" (fit) y otro método "normaliza" (transform).


# ---------------------------
# 2.1) Normalizador standard
# ---------------------------

# Definir la siguiente clase que implemente la normalización "standard", es 
# decir aquella que traslada y escala cada característica para que tenga
# media 0 y desviación típica 1. 

# En particular, definir la clase: 


class NormalizadorStandard():
    
    #Nuestra clase tendrá como elemntos la media y la desviacion típica
    def __init__(self):
        self.media=None
        self.desviacion=None
        
    #El método ajusta aplica los metodos de numpy correspondientes para calcular la media y desv
    def ajusta(self,X):
        self.media = np.mean(X, axis=0)
        self.desviacion = np.std(X, axis=0)     
    
    #Se comprueba primero que se ha ajustado y se aplica la funcion standard
    def normaliza(self,X):
        if(self.media is None or self.desviacion is None):
            raise NormalizadorNoAjustado(Exception)

        return (X - self.media) / self.desviacion
                
 


# donde el método ajusta calcula las corresondientes medias y desviaciones típicas
# de las características de X necesarias para la normalización, y el método 
# normaliza devuelve el correspondiente conjunto de datos normalizados. 

# Si se llama al método de normalización antes de ajustar el normalizador, se
# debe devolver (con raise) una excepción:

class NormalizadorNoAjustado(Exception): pass


# Por ejemplo:
    
    
# >>> normst_cancer=NormalizadorStandard()
# >>> normst_cancer.ajusta(Xe_cancer)
# >>> Xe_cancer_n=normst_cancer.normaliza(Xe_cancer)
# >>> Xv_cancer_n=normst_cancer.normaliza(Xv_cancer)
# >>> Xp_cancer_n=normst_cancer.normaliza(Xp_cancer)

# Una vez realizado esto, la media y desviación típica de Xe_cancer_n deben ser 
# 0 y 1, respectivamente. No necesariamente ocurre lo mismo con Xv_cancer_n, 
# ni con Xp_cancer_n. 


#Efectivamente la desviacion y la media nos da 1 y 0 respectivamente
# >>> np.std(Xe_cancer_n, axis=0)
# >>> np.mean(Xe_cancer_n, axis=0)

# Tal y como se dijo antes no necesariamente ocurre con este conjunto
# >>> np.mean(Xv_cancer_n, axis=0)
# >>> np.std(Xv_cancer_n, axis=0)
















# ------------------------
# 2.2) Normalizador MinMax
# ------------------------

# Hay otro tipo de normalizador, que consiste en asegurarse de que todas las
# características se desplazan y se escalan de manera que cada valor queda entre 0 y 1. 
# Es lo que se conoce como escalado MinMax

# Se pide definir la clase NormalizadorMinMax, de manera similar al normalizador 
# del apartado anterior, pero ahora implementando el escalado MinMax.

class NormalizadorMinMax():
    
    #Nuestra clase tendrá como elemntos el max y el min
    def __init__(self):
        self.min=None
        self.max=None
        
    #El método ajusta aplica los metodos de numpy correspondientes para calcular el max y min
    def ajusta(self,X):
        self.min = np.min(X, axis=0)
        self.max = np.max(X, axis=0)     

    #Se comprueba primero que se ha ajustado y se aplica la funcion minmax
    def normaliza(self,X):
        if(self.min is None or self.max is None):
            raise NormalizadorMinMaxNoAjustado(Exception)

        return (X - self.min) / (self.max - self.min)
                
 
# Si se llama al método de normalización antes de ajustar el normalizador, se
# debe devolver (con raise) una excepción:

class NormalizadorMinMaxNoAjustado(Exception): pass


# Ejemplo:

# >>> normminmax_cancer=NormalizadorMinMax()
# >>> normminmax_cancer.ajusta(Xe_cancer)
# >>> Xe_cancer_m=normminmax_cancer.normaliza(Xe_cancer)
# >>> Xv_cancer_m=normminmax_cancer.normaliza(Xv_cancer)
# >>> Xp_cancer_m=normminmax_cancer.normaliza(Xp_cancer)

# Una vez realizado esto, los máximos y mínimos de las columnas de Xe_cancer_m
#  deben ser 1 y 0, respectivamente. No necesariamente ocurre lo mismo con Xv_cancer_m,
# ni con Xp_cancer_m. 

#Efectivamente el max y el min nos da 1 y 0 respectivamente
# >>> np.min(Xe_cancer_m, axis=0)
# >>> np.max(Xe_cancer_m, axis=0)

# Tal y como se dijo antes no necesariamente ocurre con este conjunto
# >>> np.min(Xv_cancer_m, axis=0)
# >>> np.max(Xv_cancer_m, axis=0)












# ===========================================
# EJERCICIO 3: REGRESIÓN LOGÍSTICA MINI-BATCH
# ===========================================


# En este ejercicio se propone la implementación de un clasificador lineal 
# binario basado regresión logística (mini-batch), con algoritmo de entrenamiento 
# de descenso por el gradiente mini-batch (para minimizar la entropía cruzada).

from scipy.special import expit    

def sigmoide(x):
    return expit(x)

# En concreto se pide implementar una clase: 
class RegresionLogisticaMiniBatch():
    
    def __init__(self,rate=0.1,rate_decay=False,n_epochs=100,batch_tam=64):
        self.rate=rate
        self.rate_decay=rate_decay
        self.n_epochs=n_epochs
        self.batch_tam=batch_tam
        #Inicializamos los pesos con None al principio
        self.pesos=None
        #Añadimos un parametro booleano que en principio estará a False para ver si se ha entrenado antes
        self.entrenado=False
        
    def entrena(self,X,y,Xv=None,yv=None,n_epochs=100,salida_epoch=False,
                    early_stopping=False,paciencia=3):
        # Almacenamos en un atributo self.clases en una lista las clases
        self.clases=np.unique(y)
        n_epochs=self.n_epochs
        paciencia_inicial=paciencia
        self.entrenado=True
        rate_0=self.rate
                 
        #Si el valor de validacion es None se consideraría que Xv e yv son resp X e y
        if(early_stopping and ((Xv==None).all() or (yv==None).all())): 
            Xv=X; yv=y
        
        #Creamos una columna de 1s que servirán para el bias
        columna_bias_e=np.ones((X.shape[0],1))
        #Añadimos dicha columna a los datos de entrenamiento y validacion
        X=np.append(X, columna_bias_e, axis = 1)
        if(Xv is not False and Xv is not None):
            #Creamos una columna de 1s que servirán para el bias
            columna_bias_v=np.ones((Xv.shape[0],1))
            Xv=np.append(Xv, columna_bias_v, axis = 1)
        
        #Si los pesos aún no han sido inicializados, los inicializamos de amnera aleatoria
        if self.pesos == None:
            self.pesos = np.random.uniform(-1,1, X.shape[1]) #(365,30) -> (30,)  sin bias -> (31,)
            #Meterle una columna de 1s al principio para el bias
        else:
           self.pesos = self.pesos
            
        
        if(salida_epoch):
            menor_entropia=calculo_entropia_cruzada(self,X,y)
            print('Inicialmente, en entrenamiento EC:',menor_entropia,
                  ', rendimiento:',rendimiento(self,X,y))
            print('Inicialmente, en validación EC:', calculo_entropia_cruzada(self,Xv,yv),
                  ', rendimiento:',rendimiento(self,Xv,yv))
        
        # Para cada época
        for epoca in range(n_epochs):
            #Si la paciencia es igual a cero hacemos break y paramos el entrenamiento
            if(early_stopping and paciencia<=0): 
                print('PARADA TEMPRANA')
                break;
            
            #Si hay early stopping y la paciencia es mayor que cero
            if(early_stopping and paciencia>0):
                #Calculamos la entropía con la función auxiliar
                entropia_val_inicio=calculo_entropia_cruzada(self,Xv,yv)
                
                #Si es True, al inicio de cada epoch se imprime  el valor de la entropía cruzada
                if(salida_epoch): 
                    print('Epoch',epoca+1,', en entrenamiento EC:', calculo_entropia_cruzada(self,X,y),
                          ', rendimiento:',rendimiento(self,X,y))
                    print('\t\t  en validación    EC:', calculo_entropia_cruzada(self,Xv,yv),
                          ', rendimiento:',rendimiento(self,Xv,yv))
                
                if(entropia_val_inicio<menor_entropia): 
                    menor_entropia=entropia_val_inicio
        
            #Miramos si hay que actualizar el rate decay
            if(self.rate_decay):
                self.rate= (rate_0)*(1/(1+epoca)) 
            
            #DIVIDIR LOS BATCH ALEATORIAMENTE
            X_aleat,y_aleat=aleatoriedad(self, X, y)
            
            #Recomendacion usar dot!!
            
            #Para cada mini batch 
            for mini in range(int(len(X)/self.batch_tam)):
                #Calculamos los indices de inicio y final de nuestros ejemplos del mini batch
                inicio=self.batch_tam*mini
                final=self.batch_tam*(mini+1)
                
                xs=X_aleat[inicio:final] #(64,30)
                
                y_pred=y_aleat[inicio:final]   #64 predicciones (64,)
                
                y_gorro=sigmoide(np.matmul(xs,self.pesos)) #(64,30) (30,) = (64,)
                
                resta= y_pred-y_gorro #(64,) (64,) == (64,)
                
                multiplicacion=np.matmul(np.transpose(xs),resta)  #(30,64) (64,) = (30,)
                
                self.pesos=self.pesos+self.rate*multiplicacion #(30,)           
            
            
            #Si hay early stopping y la paciencia es mayor que cero
            if(early_stopping and paciencia>0):
                #Calculamos la entropía con la función auxiliar al final de la época
                entropia_val_final=calculo_entropia_cruzada(self,Xv,yv)
                
                if(entropia_val_final<menor_entropia): 
                    menor_entropia=entropia_val_final
                
                #Si la entropía cruzada al final ha disminuido
                if(entropia_val_final>menor_entropia):
                    #Bajamos la paciencia
                    paciencia=paciencia-1
                    
                #Si se ha aumentado se reinicia la paciencia 
                else:
                    paciencia=paciencia_inicial


    def clasifica_prob(self, ejemplos): 
        # Si el modelo no está entrenado devolvemos una excepción
        if self.entrenado == False:
            raise ClasificadorNoEntrenado(Exception)
        
        else:
            if(len(ejemplos.shape)==2):
                #Meterle tambien la caracteristica de la columna de 1s
                #Creamos una columna de 1s que servirán para el bias
                columna_bias=np.ones((ejemplos.shape[0],1))
                #Añadimos dicha columna a los datos de entrenamiento y validacion
                ejemplos=np.append(ejemplos, columna_bias, axis = 1)
            
            # Creamos una lista vacía para almacenar las probabilidades
            probabilidades = []  
    
            # Para cada ejemplo de la lista de ejemplos
            for ejemplo in ejemplos:  
                # Obtenemos la probabilidad de la clase positiva (índice 1)
                probabilidad = sigmoide(np.matmul(ejemplo, self.pesos))
       
                # Añadimos dicha probabilidad a la lista de probabilidades
                probabilidades.append(probabilidad)  
        
            # Convertimos la lista de probabilidades a un array numpy y lo devolvemos
            return np.array(probabilidades)  

        
    def clasifica(self, ejemplo):
        # Si el modelo no está entrenado devolvemos una excepción
        if self.entrenado == False:
            raise ClasificadorNoEntrenado(Exception)
        
        else:
            if(len(ejemplo.shape)==2 and self.pesos.shape[0]!=ejemplo.shape[1]):
                #Meterle tambien la caracteristica de la columna de 1s
                #Creamos una columna de 1s que servirán para el bias
                columna_bias=np.ones((ejemplo.shape[0],1))
                #Añadimos dicha columna a los datos de entrenamiento y validacion
                ejemplo=np.append(ejemplo, columna_bias, axis = 1)
            
            # Calculamos el producto escalar del ejemplo y el peso
            y_gorro = sigmoide(np.matmul(ejemplo, self.pesos))
            # Si el producto escalar es mayor que 0, le asignamos uno, 
            # en caso contrario le asignamos 0
            y_pred = np.where(y_gorro >= 0.5, 1, 0)
            
        return y_pred
    

# Hacemos una función auxiliar para calcular la entropía cruzada
def calculo_entropia_cruzada(self, X, y):
    
    # Multiplicamos las características X por los pesos
    y_gorro = sigmoide(np.matmul(X, self.pesos))
        
    # Calculamos la entropía cruzada utilizando np.where
    # Si y es igual a 1, selecciona np.log(y_gorro), 
    # de lo contrario selecciona np.log(1 - y_gorro)
    entropia_cruzada = np.sum(np.where(y == 1, -np.log(y_gorro),
                                               -np.log(1 - y_gorro)))
        
    return entropia_cruzada    
    
    
def aleatoriedad(self,X,y):
    indices=[i for i in range(len(X))]
    #Hacemos un random para que sea aleatorio
    np.random.shuffle(indices)
    
    #Los ejemplos del batch alaeatorios serán en el orden dado por los indices
    X_aleat=np.array([X[i] for i in indices])
    y_aleat=np.array([y[i] for i in indices])
    
    return X_aleat,y_aleat

# * El constructor tiene los siguientes argumentos de entrada:



#   + rate: si rate_decay es False, rate es la tasa de aprendizaje fija usada
#     durante todo el aprendizaje. Si rate_decay es True, rate es la
#     tasa de aprendizaje inicial. Su valor por defecto es 0.1.

#   + rate_decay, indica si la tasa de aprendizaje debe disminuir en
#     cada epoch. En concreto, si rate_decay es True, la tasa de
#     aprendizaje que se usa en el n-ésimo epoch se debe de calcular
#     con la siguiente fórmula: 
#        rate_n= (rate_0)*(1/(1+n)) 
#     donde n es el número de epoch, y rate_0 es la cantidad introducida
#     en el parámetro rate anterior. Su valor por defecto es False. 
#  
#   + batch_tam: tamaño de minibatch


# * El método entrena tiene como argumentos de entrada:
#   
#     +  Dos arrays numpy X e y, con los datos del conjunto de entrenamiento 
#        y su clasificación esperada, respectivamente. Las dos clases del problema 
#        son las que aparecen en el array y, y se deben almacenar en un atributo 
#        self.clases en una lista. La clase que se considera positiva es la que 
#        aparece en segundo lugar en esa lista.
#     
#     + Otros dos arrays Xv,yv, con los datos del conjunto de  validación, que se 
#       usarán en el caso de activar el parámetro early_stopping. Si son None (valor 
#       por defecto), se supone que en el caso de que early_stopping se active, se 
#       consideraría que Xv e yv son resp. X e y.

#     + n_epochs es el número máximo de epochs en el entrenamiento. 

#     + salida_epoch (False por defecto). Si es True, al inicio y durante el 
#       entrenamiento, cada epoch se imprime  el valor de la entropía cruzada 
#       del modelo respecto del conjunto de entrenamiento, y su rendimiento 
#       (proporción de aciertos). Igualmente para el conjunto de validación, si lo
#       hubiera. Esta opción puede ser útil para comprobar 
#       si el entrenamiento  efectivamente está haciendo descender la entropía
#       cruzada del modelo (recordemos que el objetivo del entrenamiento es 
#       encontrar los pesos que minimizan la entropía cruzada), y está haciendo 
#       subir el rendimiento.
# 
#     + early_stopping (booleano, False por defecto) y paciencia (entero, 3 por defecto).
#       Si early_stopping es True, dejará de entrenar cuando lleve un número de
#       epochs igual a paciencia sin disminuir la menor entropía conseguida hasta el momento
#       en el conjunto de validación 
#       NOTA: esto se suele hacer con mecanismo de  "callback" para recuperar el mejor modelo, 
#             pero por simplificar implementaremos esta versión más sencilla.  
#        



# * Método clasifica: recibe UN ARRAY de ejemplos (array numpy) y
#   devuelve el ARRAY de clases que el modelo predice para esos ejemplos. 

# * Un método clasifica_prob, que recibe UN ARRAY de ejemplos (array numpy) y
#   devuelve el ARRAY con las probabilidades que el modelo 
#   asigna a cada ejemplo de pertenecer a la clase positiva.       
    

# Si se llama a los métodos de clasificación antes de entrenar el modelo, se
# debe devolver (con raise) una excepción:

class ClasificadorNoEntrenado(Exception): pass

        
  

# RECOMENDACIONES: 


# + IMPORTANTE: Siempre que se pueda, tratar de evitar bucles for para recorrer 
#   los datos, usando en su lugar funciones de numpy. La diferencia en eficiencia
#   es muy grande. 

# + Téngase en cuenta que el cálculo de la entropía cruzada no es necesario
#   para el entrenamiento, aunque si salida_epoch o early_stopping es True,
#   entonces si es necesario su cálculo. Tenerlo en cuenta para no calcularla
#   cuando no sea necesario.     

# * Definir la función sigmoide usando la función expit de scipy.special, 
#   para evitar "warnings" por "overflow":

#   from scipy.special import expit    
#
#   def sigmoide(x):
#      return expit(x)

# * Usar np.where para definir la entropía cruzada. 

# -------------------------------------------------------------

# Ejemplo, usando los datos del cáncer de mama (los resultados pueden variar):


# >>> lr_cancer=RegresionLogisticaMiniBatch(rate=0.1,rate_decay=True)

# >>> lr_cancer.entrena(Xe_cancer_n,ye_cancer,Xv_cancer,yv_cancer)

# >>> lr_cancer.clasifica(Xp_cancer_n[24:27])
# array([0, 1, 0])   # Predicción para los ejemplos 24,25 y 26 

# >>> yp_cancer[24:27]
# array([0, 1, 0])   # La predicción anterior coincide con los valores esperado para esos ejemplos

# >>> lr_cancer.clasifica_prob(Xp_cancer_n[24:27])
# array([7.44297196e-17, 9.99999477e-01, 1.98547117e-18])



# Para calcular el rendimiento de un clasificador sobre un conjunto de ejemplos, usar la 
# siguiente función:
    
def rendimiento(clasif,X,y):
    return sum(clasif.clasifica(X)==y)/y.shape[0]

# Por ejemplo, los rendimientos sobre los datos (normalizados) del cáncer:
    
# >>> rendimiento(lr_cancer,Xe_cancer_n,ye_cancer)
# 0.9824561403508771

# >>> rendimiento(lr_cancer,Xp_cancer_n,yp_cancer)
# 0.9734513274336283




# Ejemplo con salida_epoch y early_stopping:

# >>> lr_cancer=RegresionLogisticaMiniBatch(rate=0.1,rate_decay=True)

# >>> lr_cancer.entrena(Xe_cancer_n,ye_cancer,Xv_cancer_n,yv_cancer,salida_epoch=True,early_stopping=True)

# Inicialmente, en entrenamiento EC: 155.686323940485, rendimiento: 0.873972602739726.
# Inicialmente, en validación    EC: 43.38533009881579, rendimiento: 0.8461538461538461.
# Epoch 1, en entrenamiento EC: 32.7750241863029, rendimiento: 0.9753424657534246.
#          en validación    EC: 8.4952918658522,  rendimiento: 0.978021978021978.
# Epoch 2, en entrenamiento EC: 28.0583715052223, rendimiento: 0.9780821917808219.
#          en validación    EC: 8.665719133490596, rendimiento: 0.967032967032967.
# Epoch 3, en entrenamiento EC: 26.857182744289368, rendimiento: 0.9780821917808219.
#          en validación    EC: 8.09511082759361, rendimiento: 0.978021978021978.
# Epoch 4, en entrenamiento EC: 26.120803184993328, rendimiento: 0.9780821917808219.
#          en validación    EC: 8.327991940213478, rendimiento: 0.967032967032967.
# Epoch 5, en entrenamiento EC: 25.66005010760342, rendimiento: 0.9808219178082191.
#          en validación    EC: 8.376171724729662, rendimiento: 0.967032967032967.
# Epoch 6, en entrenamiento EC: 25.329200890122557, rendimiento: 0.9808219178082191.
#          en validación    EC: 8.408704771704937, rendimiento: 0.967032967032967.
# PARADA TEMPRANA

# Nótese que para en el epoch 6 ya que desde la entropía cruzada obtenida en el epoch 3 
# sobre el conjunto de validación, ésta no se ha mejorado. 


# -----------------------------------------------------------------





'''
PREGUNTAR:
    -Nos salen menos imagenes de 28x28 en el conjunto de train, deben salir 5000 de 28x28
        pero solo nos salen 5000 si es 27x28
'''


















# ------------------------------------------------------------------------------





# =================================================
# EJERCICIO 4: IMPLEMENTACIÓN DE VALIDACIÓN CRUZADA
# =================================================



# Este ejercicio puede servir para el ajuste de parámetros en los ejercicios posteriores, 
# pero si no se realiza, se podrían ajustar siguiendo el método "holdout" 
# implementado en el ejercicio 1


# Definir una función: 

#  rendimiento_validacion_cruzada(clase_clasificador,params,X,y,Xv=None,yv=None,n=5)

# que devuelve el rendimiento medio de un clasificador, mediante la técnica de
# validación cruzada con n particiones. Los arrays X e y son los datos y la
# clasificación esperada, respectivamente. El argumento clase_clasificador es
# el nombre de la clase que implementa el clasificador (como por ejemplo 
# la clase RegresionLogisticaMiniBatch). El argumento params es
# un diccionario cuyas claves son nombres de parámetros del constructor del
# clasificador y los valores asociados a esas claves son los valores de esos
# parámetros para llamar al constructor.

# INDICACIÓN: para usar params al llamar al constructor del clasificador, usar
# clase_clasificador(**params)  

def rendimiento_validacion_cruzada(clase_clasificador,params,X,y,Xv=None,yv=None,n=5,salida_epoch=False):
    #Vemos de que tamaño será los pliegues 
    tam_k_partes=int(len(X)/n)
    
    #Hacemos el conjunto aleatorio y estratificado
    lista_X,lista_y=estratifica(X,y,n,tam_k_partes)
    rendimientos=[]
    #Por cada partición
    for k in range(n):
        #Cogemos como test el elemento de la lista de las particiones que toque
        test_X=np.array(lista_X[k])
        test_y=np.array(lista_y[k])
        
        #Inicializamos la lista de elementos de entrenamiento
        entrenamiento_X=[]
        entrenamiento_y=[]
        
        #Vamos iterando por los valores de n y e indexando la particion si el valor no es k
        for i in range(n):
            if (i!=k):
                entrenamiento_X.extend(lista_X[i]) 
                entrenamiento_y.extend(lista_y[i]) 
                
        #Lo pasamos a array para utilizarlo
        entrenamiento_X=np.array(entrenamiento_X)
        entrenamiento_y=np.array(entrenamiento_y)
        
        #Hacemos la clase con los parametros indicados
        clase= clase_clasificador(**params) 
        #Entrenamos la clase
        
        #Si queremos que saque por pantalla le ponemos early stooping y salida_epoch a True
        if(salida_epoch):
            clase.entrena(entrenamiento_X,entrenamiento_y,test_X,test_y,salida_epoch=True,early_stopping=True)
        #En otro caso entrenamos con esos parametros pro defecto (False)
        else:
            clase.entrena(entrenamiento_X,entrenamiento_y,test_X,test_y)
            
        #Calculamos el rendimiento para el resultado
        rendimientos.append(rendimiento(clase,test_X,test_y))
        
        print('Partición:',k+1,'. Rendimiento:',rendimiento(clase,test_X,test_y))
        
        #MEDIA DEL RENDIMIENTO SOBRE EL TEST EN CADA PARTICION
    print(np.mean(rendimientos))
    return clase
        
#Función auxiliar encargada de calcular una lista con todas las particiones hechas aleatorias
def estratifica(X,y,n,tam):
    indices=[i for i in range(len(X))]
    #Hacemos un random para que sea aleatorio
    np.random.shuffle(indices)
    
    #Inicializamos las listas vacias
    lista_X=[]
    lista_y=[]
    
    #Por cada partición    
    for i in range(n):
        #Calculamos los indices de inicio y final de nuestros ejemplos 
        inicio=tam*i
        final=tam*(i+1)
        
        #Estratificamos y ordenamos los indices de la partición
        indices_part=indices[inicio:final]
        indices_part.sort()
        
        #Escogemos los ejemplos de dichos indices
        X_part=[X[i] for i in indices_part]
        y_part=[y[i] for i in indices_part]
        #Lo añadimos a nuestras listas solucion
        lista_X.append(X_part)
        lista_y.append(y_part) 
        
    return lista_X,lista_y
    
                                   
# ------------------------------------------------------------------------------
# Ejemplo:
# --------
# Lo que sigue es un ejemplo de cómo podríamos usar esta función para
# ajustar el valor de algún parámetro. En este caso aplicamos validación
# cruzada, con n=5, en el conjunto de datos del cancer, para estimar cómo de
# bueno es el valor batch_tam=16 con rate_decay en regresión logística mini_batch.
# Usando la función que se pide sería (nótese que debido a la aleatoriedad, 
# no tiene por qué coincidir el resultado):

# >>> rendimiento_validacion_cruzada(RegresionLogisticaMiniBatch,
#                                {"batch_tam":16,"rate":0.01,"rate_decay":True},
#                                 Xe_cancer_n,ye_cancer,n=5)

# Partición: 1. Rendimiento:0.9863013698630136
# Partición: 2. Rendimiento:0.958904109589041
# Partición: 3. Rendimiento:0.9863013698630136
# Partición: 4. Rendimiento:0.9726027397260274
# Partición: 5. Rendimiento:0.9315068493150684
# >>> 0.9671232876712328




# El resultado es la media de rendimientos obtenidos entrenando cada vez con
# todas las particiones menos una, y probando el rendimiento con la parte que
# se ha dejado fuera. Las particiones DEBEN SER ALEATORIAS Y ESTRATIFICADAS. 
 
# Si decidimos que es es un buen rendimiento (comparando con lo obtenido para
# otros valores de esos parámetros), finalmente entrenaríamos con el conjunto de
# entrenamiento completo:

# >>> lr16=RegresionLogisticaMiniBatch(batch_tam=16,rate=0.01,rate_decay=True)
# >>> lr16.entrena(Xe_cancer_n,ye_cancer)

# Y daríamos como estimación final el rendimiento en el conjunto de prueba, que
# hasta ahora no hemos usado:
# >>> rendimiento(lr16,Xp_cancer_n,yp_cancer)
# 0.9646017699115044

#------------------------------------------------------------------------------

















# ===================================================
# EJERCICIO 5: APLICANDO LOS CLASIFICADORES BINARIOS
# ===================================================

# Usando la regeresión logística implementada en el ejercicio 2, obtener clasificadores 
# con el mejor rendimiento posible para los siguientes conjunto de datos:

# - Votos de congresistas US
# - Cáncer de mama 
# - Críticas de películas en IMDB

# Ajustar los parámetros (tasa, rate_decay, batch_tam) para mejorar el rendimiento 
# (no es necesario ser muy exhaustivo, tan solo probar algunas combinaciones). 
# Si se ha hecho el ejercicio 4, usar validación cruzada para el ajuste 
# (si no, usar el "holdout" del ejercicio 1). 

# Mostrar el proceso realizado en cada caso, y los rendimientos finales obtenidos
# sobre un conjunto de prueba.     

# Mostrar también, para cada conjunto de datos, un ejemplo con salida_epoch, 
# en el que se vea cómo desciende la entropía cruzada y aumenta el 
# rendimiento durante un entrenamiento.     

# ----------------------------

def preEntrenado(X,y,normalizador,porcP,porcV,Xp=None,yp=None):
    #Si no teiene conjunto de prueba
    if((Xp is None) or (yp is None)):
        #Divide entrenamiento-validacion y prueba
        Xev,Xp,yev,yp=particion_entr_prueba(X, y,porcP)
        
        #Divide el conjunto anterior entrenamiento-validacion en dos uno para entrenamiento y otra para val
        Xe,Xv,ye,yv=particion_entr_prueba(Xev, yev,porcV)
        
    #Si ya tiene conjunto de prueba
    else:
        #Divide el conjunto anterior entrenamiento-validacion en dos uno para entrenamiento y otra para val
        Xe,Xv,ye,yv=particion_entr_prueba(X, y,porcV)
   
    
    #Ajustamos el normalizador con los datos de entrenamiento
    normalizador.ajusta(Xe)
    
    #Aplizamos a los conkjuntos
    Xe_n=normalizador.normaliza(Xe)
    Xp_n=normalizador.normaliza(Xp)
    Xv_n=normalizador.normaliza(Xv)
    
    return Xe_n,ye,Xp_n,yp,Xv_n,yv
    
def entrena_saca_rend(X,y,Xp,yp,Xv,yv,clasificador,params,n):    
    #Calcula rendimiento con los parametros pasados
    #Utilzando la funcion rendimiento_validacion_cruzada
    clase=rendimiento_validacion_cruzada(clasificador,params,X,y,Xv,yv,n)

    #Saco rendimiento sobre el de prueba utilizando el rendimiento(clase,Xp,yp)
    print('Rendimiento prueba:',rendimiento(clase,Xp,yp))
    

    #haz ejemplo con salida epoch
    #Utilzando la funcion rendimiento_validacion_cruzada
    clase=rendimiento_validacion_cruzada(clasificador,params,X,y,Xv,yv,n,salida_epoch=True)
    
    #Saco rendimiento sobre el de prueba utilizando el rendimiento(clase,Xp,yp)
    print('Rendimiento prueba:',rendimiento(clase,Xp,yp))
    return rendimiento(clase,Xp,yp)
    

#Definimos tres diccionarios de parametros para hacer las pruebas
param1={"batch_tam":24,"n_epochs":25,"rate":0.013,"rate_decay":True}
param2={"batch_tam":24,"n_epochs":35,"rate":0.013,"rate_decay":True}
param3={"batch_tam":24,"n_epochs":45,"rate":0.013,"rate_decay":True}

#Tres numeros de particiones diferentes
n1=5;n2=7;n3=4

#Normalizamos y calculamos los conjuntos de entrenamiento,prueba y validacion
# >>> Xe_cancer_n,ye_cancer,Xp_cancer_n,yp_cancer,Xv_cancer_n,yv_cancer=preEntrenado(X_cancer,y_cancer,NormalizadorStandard(),1/3,0.2)
# >>> Xe_votos_n,ye_votos,Xp_votos_n,yp_votos,Xv_votos_n,yv_votos=preEntrenado(X_votos,y_votos,NormalizadorStandard(),1/3,0.2)
# >>> Xe_imdb_n,ye_imdb,Xp_imdb_n,yp_imdb,Xv_imdb_n,yv_imdb=preEntrenado(X_train_imdb,y_train_imdb,NormalizadorStandard(),0.2,0.2,X_test_imdb,y_test_imdb)

#Sacamos el rendimiento con los parametros del diccionario y numero de particiones escogidas
# >>> entrena_saca_rend(Xe_cancer_n,ye_cancer,Xp_cancer_n,yp_cancer,Xv_cancer_n,yv_cancer,RegresionLogisticaMiniBatch,param1,n1)
# >>> entrena_saca_rend(Xe_votos_n,ye_votos,Xp_votos_n,yp_votos,Xv_votos_n,yv_votos,RegresionLogisticaMiniBatch, param2,n2)
# >>> entrena_saca_rend( Xe_imdb_n,ye_imdb,Xp_imdb_n,yp_imdb,Xv_imdb_n,yv_imdb,RegresionLogisticaMiniBatch,param3,n3)





# =====================================================
# EJERCICIO 6: CLASIFICACIÓN MULTICLASE CON ONE vs REST
# =====================================================

# Se pide implementar un algoritmo de regresión logística para problemas de
# clasificación en los que hay más de dos clases, usando  la técnica One vs Rest. 


#  Para ello, implementar una clase  RL_OvR con la siguiente estructura, y que 
#  implemente un clasificador OvR (one versus rest) usando como base el
#  clasificador binario RegresionLogisticaMiniBatch


# class RL_OvR():

#     def __init__(self,rate=0.1,rate_decay=False,
#                   batch_tam=64):

#        ......

#     def entrena(self,X,y,n_epochs=100,salida_epoch=False):

#        .......

#     def clasifica(self,ejemplos):

#        ......
            



#  Los parámetros de los métodos significan lo mismo que en el apartado
#  anterior, aunque ahora referido a cada uno de los k entrenamientos a 
#  realizar (donde k es el número de clases).
#  Por simplificar, supondremos que no hay conjunto de validación ni parada
#  temprana.  

 

#  Un ejemplo de sesión, con el problema del iris:


# --------------------------------------------------------------------
# >>> Xe_iris,Xp_iris,ye_iris,yp_iris=particion_entr_prueba(X_iris,y_iris)

# >>> rl_iris_ovr=RL_OvR(rate=0.001,batch_tam=8)

# >>> rl_iris_ovr.entrena(Xe_iris,ye_iris)

# >>> rendimiento(rl_iris_ovr,Xe_iris,ye_iris)
# 0.8333333333333334

# >>> rendimiento(rl_iris_ovr,Xp_iris,yp_iris)
# >>> 0.9
# --------------------------------------------------------------------










            
# --------------------------------
class RL_OvR():

    def __init__(self,rate=0.1,rate_decay=False,
                  batch_tam=64,n_epochs=100):
        self.rate=rate
        self.rate_decay=rate_decay
        self.batch_tam=batch_tam
        self.entrenado=False
        self.n_epochs=n_epochs
        self.clasificadores=[]

    def entrena(self,X,y,n_epochs=100,salida_epoch=False):
        #Lo pinemos como entrenado
        self.entrenado=True
        #Guardamos las clases en un array
        self.clases=np.unique(y)
        
        
        #Por cada clase (clase_actual)
        for clase in self.clases:
            #Modificamos las clasificaciones para que la clase actual sea la positiva y el resto 0
            y_modificado=np.where(y==clase, 1,0)
            #Creamos el clasificador
            clasificador=RegresionLogisticaMiniBatch(rate=self.rate,rate_decay=self.rate_decay,
                                                     batch_tam=self.batch_tam,n_epochs=self.n_epochs)
            
            #Entrenas con la y ya modificada para que la clase actual sea la positiva
            clasificador.entrena(X,y_modificado,salida_epoch)
            
            #Guardo el clasificador para utilziarlo en la funcion clasifica
            self.clasificadores.append(clasificador)
        

    def clasifica(self,ejemplos):
        predicciones=[]
  
        #Por cada ejemplo de los ejemplos
        for i in range(len(ejemplos)):
            valores=[]
            #Aplicas cada clasificador y devuelves aquella clase con el mayor valor
            for clasificador in self.clasificadores:
                
                #Calculamos la probabilidad de pertenecer a esa clase                
                valor=clasificador.clasifica_prob(ejemplos)
                
                #Lo indexamos a la lista de las probabilidades
                valores.append(valor[i])
                
            
            #Nos quedamos con el indice (que indica el numero de la clase) del mayor valor de las porbabilidade
            predicciones.append(np.argmax(valores))
        return predicciones
            



# =================================
# EJERCICIO 7: CODIFICACIÓN ONE-HOT
# =================================


# Los conjuntos de datos en los que algunos atributos son categóricos (es decir,
# sus posibles valores no son numéricos, o aunque sean numéricos no hay una 
# relación natural de orden entre los valores) no se pueden usar directamente
# con los modelos de regresión logística, o con redes neuronales, por ejemplo.

# En ese caso es usual transformar previamente los datos usando la llamada
# "codificación one-hot". Básicamente, cada columna se reemplaza por k columnas
# en los que los valores psoibles son 0 o 1, y donde k es el número de posibles 
# valores del atributo. El valor i-ésimo del atributo se convierte en k valores
# (0 ...0 1 0 ...0 ) donde todas las posiciones son cero excepto la i-ésima.  

# Por ejemplo, si un atributo tiene tres posibles valores "a", "b" y "c", ese atributo 
# se reemplazaría por tres atributos binarios, con la siguiente codificación:
# "a" --> (1 0 0)
# "b" --> (0 1 0)
# "c" --> (0 0 1)    

# Definir una función:    
    
#     codifica_one_hot(X) 

# que recibe un conjunto de datos X (array de numpy) y devuelve un array de numpy
# resultante de aplicar la codificación one-hot a X.Por simplificar supondremos 
# que el array de entrada tiene todos sus atributos categóricos, y que por tanto 
# hay que codificarlos todos.

# Aplicar la función para obtener una codificación one-hot de los datos sobre
# concesión de prestamo bancario.     
'''
>>> Xc=np.array([["a",1,"c","x"],
                  ["b",2,"c","y"],
                  ["c",1,"d","x"],
                  ["a",2,"d","z"],
                  ["c",1,"e","y"],
                  ["c",2,"f","y"]])
'''   
# >>> codifica_one_hot(Xc)
# 
# array([[1., 0., 0., 1., 0., 1., 0., 0., 0., 1., 0., 0.],
#        [0., 1., 0., 0., 1., 1., 0., 0., 0., 0., 1., 0.],
#        [0., 0., 1., 1., 0., 0., 1., 0., 0., 1., 0., 0.],
#        [1., 0., 0., 0., 1., 0., 1., 0., 0., 0., 0., 1.],
#        [0., 0., 1., 1., 0., 0., 0., 1., 0., 0., 1., 0.],
#        [0., 0., 1., 0., 1., 0., 0., 0., 1., 0., 1., 0.]])

# En este ejemplo, cada columna del conjuto de datos original se transforma en:
#   * Columna 0 ---> Columnas 0,1,2
#   * Columna 1 ---> Columnas 3,4
#   * Columna 2 ---> Columnas 5,6,7,8
#   * Columna 3 ---> Columnas 9, 10,11     

    
  

# -------- 

def codifica_one_hot(X):
    #Transponemos la matriz para utilziarla mas tarde
    X_tras=np.transpose(X)
    #Creamos lo que será nuestro resultado incialmente vacio
    X_final=[]
    
    #Por cada columna de nuestro array de entrada
    for columna in range(len(X[0])):
        #Vemos que valores puede tomar en dicha columna
        valores=list(np.unique( X_tras[columna]))
        
        #Cogemos la columna 
        filaArray=X_tras[columna]
        
        #Por cada vaor que pueda tomar crearemos una columna en el array resultado
        for valor in valores:
            #Miramos si es igual el valor al de la columna, le ponemos 1 si se cumple y 0 eoc
            fila= np.where(filaArray==valor,1.,0.)
            
            #Indexamos al array final dicha columna
            X_final.append(np.array(fila))
            
    #Trasponemos de nuevo y convertimos a array
    X_final=np.transpose(np.array(X_final))
    
    return X_final
        
    


def codifica_labels(y):
    #Cogemos las clases diferentes que hay
    clases= np.unique(y)
    #Hacemos un zip con la clase y su indice
    clases_num=zip(clases,range(len(clases)))
    #Por cada clase sustituimos el valor de la clase por el indice
    for clase,i in clases_num:
        y=np.where(y==clase,i,y)
    
    #Lo transformamos a array
    y=np.array(y)
    #Como np.eere mantiene el formato inicial lo pasamos de str a int
    y_final=np.array([int(i) for i in y])
    return y_final





# =====================================================
# EJERCICIO 8: APLICACIONES DEL CLASIFICADOR MULTICLASE
# =====================================================


# ---------------------------------------------------------
# 8.1) Conjunto de datos de la concesión de crédito
# ---------------------------------------------------------

# Aplicar la implementación OvR Y one-hot de los ejercicios anteriores,
# para obtener un clasificador que aconseje la concesión, 
# estudio o no concesión de un préstamo, basado en los datos X_credito, y_credito. 

# Ajustar adecuadamente los parámetros (nuevamente, no es necesario ser demasiado 
# exhaustivo)

# ----------------------



def estudia_prestamo(X,y,codificacion=True):
    #Creamos una lista de rendimientos vacia
    rendimientos=[]
    #Cidificacmos a one hot el conjunto de datos
    if(codificacion):
        X_codificado=codifica_one_hot(X)
        y_codificado=codifica_labels(y)
        
    else:
        X_codificado=X
        y_codificado=y
    
    #Guardamos los parametros anteriormente difeinidos en una lista para recorrerla
    params=[param1,param2,param3]
    
    #Por cada parametros
    for i in range(3):
        param=params[i]
        
        #Creamos un clasificador con ellos y entrenamos
        clasif=RL_OvR(** param)
        clasif.entrena(X_codificado,y_codificado)
        
        #Le sacamos el rendimiento y guardamos en la lista de rendimientos        
        rend=rendimiento(clasif,X_codificado,y_codificado)        
        rendimientos.append(rend)
    
    #De la lista de rendimientos cogemos el indice del mayor valor
    mejor= np.argmax(rendimientos)
    #Para quedarnos con dicho parametros
    mejor_parametro=params[mejor]
    
    #Creamos el clasficador y entrenamos
    clasificador=RL_OvR(** mejor_parametro)    
    clasificador.entrena( X_codificado,y_codificado)
    
    print(rendimiento(clasificador, X_codificado,y_codificado))
    #Devolvemos el clasificador
    return clasificador
    



# >>> Rl_credito= estudia_prestamo(X_credito,y_credito,True)



# ---------------------------------------------------------
# 8.2) Clasificación de imágenes de dígitos escritos a mano
# ---------------------------------------------------------


#  Aplicar la implementación OvR anterior, para obtener un
#  clasificador que prediga el dígito que se ha escrito a mano y que se
#  dispone en forma de imagen pixelada, a partir de los datos que están en el
#  archivo digidata.zip que se suministra.  Cada imagen viene dada por 28x28
#  píxeles, y cada pixel vendrá representado por un caracter "espacio en
#  blanco" (pixel blanco) o los caracteres "+" (borde del dígito) o "#"
#  (interior del dígito). En nuestro caso trataremos ambos como un pixel negro
#  (es decir, no distinguiremos entre el borde y el interior). En cada
#  conjunto las imágenes vienen todas seguidas en un fichero de texto, y las
#  clasificaciones de cada imagen (es decir, el número que representan) vienen
#  en un fichero aparte, en el mismo orden. Será necesario, por tanto, definir
#  funciones python que lean esos ficheros y obtengan los datos en el mismo
#  formato numpy en el que los necesita el clasificador. 

#  Los datos están ya separados en entrenamiento, validación y prueba. En este
#  caso concreto, NO USAR VALIDACIÓN CRUZADA para ajustar, ya que podría
#  tardar bastante (basta con ajustar comparando el rendimiento en
#  validación). Si el tiempo de cómputo en el entrenamiento no permite
#  terminar en un tiempo razonable, usar menos ejemplos de cada conjunto.

# Ajustar los parámetros de tamaño de batch, tasa de aprendizaje y
# rate_decay para tratar de obtener un rendimiento aceptable (por encima del
# 75% de aciertos sobre test). 


# --------------------------------------------------------------------------
   
def lectura_imagenes(ruta,labels=False):
    imagenes=[]
    #Leemos el fichero
    with open (ruta,'r') as file:
        #Creamos un numero inicialmente vacio que contendra toda las lineas de un unico numero
        numero= []
        cont=27
        #Por cada linea del fichero 
        for linea in file:
            
            #print(linea)
            #Le quitamos el salto de linea
            linea=linea[:-1]
            #Si el fichero es de etiquetas 
            if(labels):
                #Pasamos la linea a un numero y lo metemos en nuestra lista solucion
                imagenes.append(int(linea))
                
            #Eoc estaremos ante un fichero de imagenes
            else:
                #Si todavia quedan lineas del numero 
                if(not cont==0):
                    #Añadimos esa linea al numero
                    numero.append(linea)
                    cont=cont-1
                
                #Cuando estamos en la ultima linea
                else: 
                    #Añadimos esa linea al numero
                    numero.append(linea)
                    #Transformamos esas lineas a un array de 1s y 0s
                    imagen=transforma_imagen(numero)
                    #Lo añadimos ya transformado a la lista solucion
                    imagenes.append(imagen)
                    #Ponemos la lista del numero vacia de nuevo 
                    numero=[]
                    cont=27
                    
    #Transformamos la lista solucion en un array
    imagenes=np.array(imagenes)
    
    #Si el conjunto es de imagenes (no es de labels)
    if(not labels):
        #Cambiamos el shape de cada imagen de (28,28) a (784), es decir, la aplanamos
        imagenes=np.reshape(imagenes,(imagenes.shape[0],imagenes.shape[1]*imagenes.shape[2]))
    
    return imagenes
    
#METER ETIQUETAS EB OTRA FUNCION

def transforma_imagen(numero):
    numero_res=[]
    #Por cada linea de nuestro numero
    for linea in numero:
        #Creamos una linea solucion donde alñadir los 0s y 1s
        linea_res=[]
        
        #Por cada pixxel de la linea
        for pixel in linea:
            
            #Si es un espacio en blanco le ponemos un 0
            if(pixel==" "):
                linea_res.append(0)
            #En otro caso le ponemos un 1
            else:
                linea_res.append(1)
                
        #Pasamos la linea a un array y lo añadimos a la solucion
        linea_res=np.array(linea_res)
        numero_res.append(linea_res)
    
    #Pasamos la solucion a un array y lo devolvemos
    numero_res=np.array(numero_res)
    
    return numero_res
    




# >>> train_img=lectura_imagenes('datos/digitdata/trainingimages')
# >>> test_img=lectura_imagenes('datos/digitdata/testimages')
# >>> y_train=lectura_imagenes('datos/digitdata/traininglabels',True)
# >>> y_test=lectura_imagenes('datos/digitdata/testlabels',True)
# >>> val_img=lectura_imagenes('datos/digitdata/validationimages')
# >>> y_val=lectura_imagenes('datos/digitdata/validationlabels',True)




def mejor_clasificador_digitos(Xe,ye,Xp,yp,Xv,yv):
    rendimientos=[]
    #Guardamos los parametros anteriormente difeinidos en una lista para recorrerla
    params=[param1,param2,param3]
    
    #Por cada parametros
    for i in range(3):
        param=params[i]
        
        #Creamos un clasificador con ellos y entrenamos
        clasif=RL_OvR(** param)
        clasif.entrena(Xe,ye)
        #Le sacamos el rendimiento y guardamos en la lista de rendimientos
        rend=rendimiento(clasif,Xv,yv) 
        
        print(rend)
        rendimientos.append(rend)
    
    #De la lista de rendimientos cogemos el indice del mayor valor
    mejor= np.argmax(rendimientos)   
    
    print('Maximo rendimiento:',max(rendimientos))
    
    #Para quedarnos con dicho parametros
    mejor_parametro=params[mejor]
    
    #Creamos el clasficador y entrenamos
    clasificador=RL_OvR(** mejor_parametro)    
    clasificador.entrena(Xe,ye)

    print(rendimiento(clasificador,Xp,yp))
    #Devolvemos el clasificador
    return clasificador


#Definimos tres diccionarios de parametros para hacer las pruebas
param1={"batch_tam":64,"n_epochs":45, "rate":0.1,"rate_decay":True}
param2={"batch_tam":64,"n_epochs":120, "rate":0.073,"rate_decay":True}
param3={"batch_tam":32,"n_epochs":120,"rate":0.1,"rate_decay":True}



# >>> clasif_digits= mejor_clasificador_digitos(train_img,y_train,test_img,y_test,val_img,y_val)







# =========================================================================
# EJERCICIO OPCIONAL PARA SUBIR NOTA: 
#    CLASIFICACIÓN MULTICLASE CON REGRESIÓN LOGÍSTICA MULTINOMIAL
# =========================================================================


#  Se pide implementar un clasificador para regresión
#  multinomial logística con softmax (VERSIÓN MINIBATCH), descrito en las 
#  diapositivas 55 a 57 del tema de "Complementos de Aprendizaje Automático". 

# class RL_Multinomial():

#     def __init__(self,rate=0.1,rate_decay=False,
#                   batch_tam=64):

#        ......

#     def entrena(self,X,y,n_epochs=100,salida_epoch=False):

#        .......

#     def clasifica_prob(self,ejemplos):

#        ......
 

#     def clasifica(self,ejemplos):

#        ......
   

 
# Los parámetros tiene el mismo significado que en el ejercicio 7 de OvR. 

# En eset caso, tiene sentido definir un clasifica_prob, ya que la función
# softmax nos va a devolver una distribución de probabilidad de pertenecia 
# a las distintas clases. 


# NOTA 1: De nuevo, es muy importante para la eficiencia usar numpy para evitar
#         el uso de bucles for convencionales.  

# NOTA 2: Se recomienda usar la función softmax de scipy.special: 

    # from scipy.special import softmax   
#

    
# --------------------------------------------------------------------

# Ejemplo:
# >>> Xe_iris,Xp_iris,ye_iris,yp_iris=particion_entr_prueba(X_iris,y_iris)

# >>> rl_iris_m=RL_Multinomial(rate=0.001,batch_tam=8)

# >>> rl_iris_m.entrena(Xe_iris,ye_iris,n_epochs=50)

# >>> rendimiento(rl_iris_m,Xe_iris,ye_iris)
# 0.9732142857142857

# >>> rendimiento(rl_iris_m,Xp_iris,yp_iris)
# >>> 0.9736842105263158
# --------------------------------------------------------------------

# --------------- 
from scipy.special import softmax   

class RL_Multinomial():

    def __init__(self,rate=0.1,rate_decay=False,n_epochs=100,batch_tam=64):
       self.rate=rate
       self.rate_decay=rate_decay
       self.batch_tam=batch_tam
       self.entrenado=False
       self.pesos=[]
       self.n_epochs=n_epochs
    
    def entrena(self,X,y,n_epochs=100,salida_epoch=False):
        #Vemos cuales son las clases en las labels
        self.clases=np.unique(y)
        #Nos quedamos el valor incial del rate
        rate_0=self.rate
        #Lo ponemos como entrenado
        self.entrenado=True
        
        #Creamos una columna de 1s que servirán para el bias
        columna_bias_e=np.ones((X.shape[0],1))
        #Añadimos dicha columna a los datos de entrenamiento 
        X=np.append(X, columna_bias_e, axis = 1)
        
        #Si los pesos aún no han sido inicializados, los inicializamos de amnera aleatoria
        if self.pesos == []:            
            #Por cada clase tendremos un vector de pesos
            for _ in self.clases:
                self.pesos.append(np.random.uniform(-1,1, X.shape[1]))
        
        #Por cada epoca
        for epoca in range(n_epochs):
            
            #Miramos si hay que actualizar el rate decay y lo actualizamos si es asi
            if(self.rate_decay):
                self.rate= (rate_0)*(1/(1+epoca)) 
            
            #ividimos los batch de manera aleatoria
            X_aleat,y_aleat=aleatoriedad(self, X, y)
            
            #Transformamos la y para que sea 1 si es de la clase y 0 en otro caso
            y_aleat=codifica_multinomial(self,y_aleat)
            
            
            #Para cada mini batch 
            for mini in range(int(len(X)/self.batch_tam)):

                    #Calculamos los indices de inicio y final de nuestros ejemplos del mini batch
                    inicio=self.batch_tam*mini
                    final=self.batch_tam*(mini+1)
                    
                    xs=X_aleat[inicio:final] #Cogemos los ejemplos del mini
                    
                    y_pred=y_aleat[inicio:final]   #Cogemos el valor de la y de los ejemplos
                    
                    mult=[np.matmul(xs,self.pesos[clase] )for clase in self.clases]

                    y_gorro=softmax(mult) #Claculamos la prediccion softmax
                    
                    #print('Y gorro',np.transpose(y_gorro))

                    resta= y_pred-np.transpose(y_gorro) #Restamos lo predecido con el valor real
           
                    multiplicacion=np.matmul(np.transpose(resta),xs)  #Hayamos la multiplicacion
                    self.pesos=self.pesos+self.rate*multiplicacion #Actualizamos el vector de pesos         
                    
                        
   
        
   
    def clasifica_prob(self, ejemplos): 
        # Si el modelo no está entrenado devolvemos una excepción
        if self.entrenado == False:
            raise ClasificadorNoEntrenado(Exception)
            
        else:
            print(self.clases)
            # Creamos una lista vacía para almacenar las probabilidades
            probabilidades = []  
            # Para cada ejemplo de la lista de ejemplos
            for ejemplo in ejemplos:  

                multiplicacion=[np.matmul(ejemplo,self.pesos[clase] )for clase in self.clases]
                print(multiplicacion)
                # Obtenemos la probabilidad de la clase positiva (índice 1)
                probabilidad = softmax(multiplicacion)
                
                
                probabilidades.append(probabilidad)
                
            # Convertimos la lista de probabilidades a un array numpy y lo devolvemos
            #print('Probabilidades de los ejemplos',ejemplos.shape,':\n',probabilidades)
        
            return np.array(probabilidades)  
    
        
    def clasifica(self, ejemplos):
        y_pred=[]
        # Si el modelo no está entrenado devolvemos una excepción
        if self.entrenado == False:
            raise ClasificadorNoEntrenado(Exception)
        
        else:
            
            if(self.pesos[0].shape != ejemplos.shape):
                # Creamos una columna de 1s que servirán para el bias
                columna_bias = np.ones((ejemplos.shape[0], 1))
                # Añadimos dicha columna a los datos de entrenamiento y validacion
                ejemplos = np.append(ejemplos, columna_bias, axis = 1)

            y_gorro=self.clasifica_prob(ejemplos)
            
            for ejemplo in y_gorro:

                #print('Prediccion de un ejemplo',ejemplo,':\n',y_gorro),+
                y_pred.append(np.argmax(ejemplo))
        y_pred=np.array(y_pred)
        print(y_pred)
        return y_pred



def codifica_multinomial(self,y):
    #Creamos lo que será nuestro resultado incialmente vacio
    X_final=[]
    
    for i in range(len(y)):
        clase=y[i]
        valores=np.zeros(len(self.clases),)
        valores[clase]=1
        X_final.append(valores)
    X_final=np.array(X_final)
    
    return X_final



