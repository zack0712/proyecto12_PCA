import streamlit as st
import pandas as pd 
from PIL import Image

image1 = Image.open('pca_2componentes.jpg')
image2 = Image.open('pca_3componentes.jpg')
image3 = Image.open('Indice.jpg')
image4 = Image.open('2cluster_pca.jpg')
image5 = Image.open('8cluster_pca.jpg')

st.title('**PCA y Cluster con Kmeans, Indice de Felicidad Banco Mundial**')
st.write('''
Este proyecto busca desarrollar algoritmos de reducción de dimensionalidad como PCA o también algoritmos de clasificación \
en grupos como Kmeans. Lo anterior sobre un conjunto de variables extraídas usando la API del banco mundial, puesto que se \
esta buscando construir un índice de felicidad de los países.Este proyecto busca desarrollar algoritmos de reducción de \
dimensionalidad como PCA o también algoritmos de clasificación en grupos como Kmeans. Lo anterior sobre un conjunto de \
variables extraídas usando la API del banco mundial, puesto que se esta buscando construir un índice de felicidad de los países.
 ''')

st.subheader('Construccion de los datos')

st.write('Los datos usados para este proyecto se construyeron usando la API del Banco Mundial, de manera que se extraen 10 variables\
 para el total de países disponibles durante el año 2019, de manera que se filtran los datos faltantes y se maneja una tabla de 11 columnas \
(incluyendo una de los nombres de países) y las otras 10 asociadas a indicadores económicos de cada país, adicionalmente la tabla cuenta después \
del filtrado de los valores nulos con 64 países. A continuación, se presenta el código que se uso para construir la base de datos: ')

datos = pd.read_csv('datos.csv')

st.code('''
variables = ['espectativa_vida','emisiones_co2','gasto_nacional','tasa_interes_real', 'gastos','crec_pib_percapita','ahorro_bruto','rentas_petroleras','ingresos_fiscales','pib_percapita']

codigos = ['SP.DYN.LE00.IN','EN.ATM.CO2E.KD.GD','NE.DAB.TOTL.ZS','FR.INR.RINR', 'GC.XPN.TOTL.GD.ZS','NY.GDP.PCAP.KD.ZG', 'SH.XPD.GHED.GD.ZS','NY.GDP.PETR.RT.ZS','GC.TAX.TOTL.GD.ZS','NY.GDP.PCAP.CD']

dic = {codigo:variable for (codigo,variable) in zip(codigos,variables)}


datos = wb.data.DataFrame(codigos[0], time=[2019], labels=True,)
datos = datos['Country'] 
for var,name in zip(codigos,variables):
  df = wb.data.DataFrame(var, time=[2019], labels=True,)
  df = df.reset_index(drop=True)
  datos = pd.merge(datos,df,how='outer',on='Country')

datos = datos.dropna().rename(columns=dic)
datos['pib_categoria'] = pd.cut(datos['pib_percapita'],bins=datos['pib_percapita'].quantile([0,0.50,0.75,1]).values,labels=[0,1,2])
datos['pib_categoria'] = datos['pib_categoria'].fillna(0)
''',language='python')

st.table(datos.head())

st.subheader('**PCA (2 y 3 Componentes)**')
st.write('''
Se usará un PCA para reducir las dimensiones del DataFrame, se analizará la varianza explicada por cada componente para determinar el numero \
optimo de dimensiones, posteriormente se desarrollará un análisis grafico del PCA para el caso de 2 componentes y el de 3 componentes. 
''')

st.code(
''' 
X = datos.drop(columns=['Country']).to_numpy()
scaler = StandardScaler()
X_scal = scaler.fit_transform(X)

pca = PCA(n_components=9)
X_pca = pca.fit_transform(X_scal)

## PCA 2 componentes 

pca = PCA(n_components=2,random_state=7)
X_pca = pca.fit_transform(X_scal)
pca2 = pd.DataFrame(X_pca,columns=['pc1','pc2'])

''',language='python')

st.image(image1)

st.code(
'''
pca = PCA(n_components=3)
X_pca = pca.fit_transform(X_scal)
pca3 = pd.DataFrame(X_pca,columns=['pc1','pc2','pc3'])
'''
)
st.image(image2)

st.subheader('Clustering con Kmeans')

st.write('''
En esta sección se busca crear grupos para visualizar la distribución de los componentes principales teniendo en cuenta categorías que se\
 asignaran por medio de un clúster Kmeans, Así mismo se tendrá en cuenta el índice de Davis Boulding en el análisis de los cluster.
'''
)

st.code('''
indices_davies = []
for i in range(2,9):
  kmeans = KMeans(n_clusters=i, random_state=777,algorithm='elkan').fit(X_scal)
  labels = kmeans.labels_
  indices_davies.append(davies_bouldin_score(X_scal, labels))
''')

st.image(image3)

st.code('''
kmeans = KMeans(n_clusters=2, random_state=777,algorithm='elkan').fit(X_scal)
pca2['labels'] = kmeans.labels_
''')

st.image(image4)

st.write('''
Observando el comportamiento del índice de Davies boundling se puede ver que el numero de clusters mas optimo es 8, de manera que a\
continuación se presentara la construcción de estos grupos. 
''')

st.code('''
kmeans = KMeans(n_clusters=8, random_state=777,algorithm='elkan').fit(X_scal)
pca2['labels'] = kmeans.labels_
''')

st.image(image5)

st.subheader('''**Concluciones Generales**''')

st.write('''
A priori se podría pensar que es difícil clasificar a los países con las variables seleccionadas, en un principio los PCA realizados \
no muestran una clasificación en grupos tan clara, esto puede deberse a la selección de las variables, aun así se debe tener en cuenta\
 el hecho de que es muy probable que por el contexto de lo que se busca clasificar haya un grupo con muchos individuos mientras que otro \
simplemente estén formados por los extremos (los países más felices o los más infelices )
'''
)

st.write('Link al notebook del proyecto:  https://github.com/zack0712/proyecto12_PCA/blob/main/Banco_Mundial_hugo.ipynb')