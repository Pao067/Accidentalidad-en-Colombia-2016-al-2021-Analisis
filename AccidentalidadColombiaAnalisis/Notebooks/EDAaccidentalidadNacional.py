
import pandas as pd
## ... Accidentalidad Vial 2017 - 2021. Transportation. Eventos de accidentalidad registrados en las vias administradas por INVIAS. Last UpdatedDecember 9, 2024
###### ANALISIS EXPLORATORIO DE LOS DATOS(EDA)  ######

### 1. CARGAR LOS DATOS
f=r"C:\AccidentalidadColombiaAnalisis\Data\Raw\Accidentalidad_Vial_2017_2021.csv"
ac = pd.read_csv(f)
ac

### 2. VISTAS RAPIDAS

pd.options.display.max_columns = None 
ac.head()
ac.tail()

# pd.set_option → Permite modificar las opciones de visualización de pandas.
# 'display.max_columns' → Es el parámetro que controla cuántas columnas se muestran al imprimir un DataFrame.
# None → Indica que no hay un límite en la cantidad de columnas que se pueden mostrar.

# Revisar todas las columnas

ac.columns

### 3. VERIFICARTIPOS DE DATOS Y VALORES NULOS
ac.info()
# RangeIndex: 18553 
# Columns: 41

### 4. DATOS DUPLICADOS
ac.duplicated().sum()

###.5. VALORES NULOS Y MANEJO DE AUSENCIAS AUSENCIAS

ac.isnull().sum()
(ac.isnull().sum()/18553)*100

## Variable Fecha_Registro

# Importante identificar fechas con valores inconsistentes

# Fecha_Registro---- aproximadamente un 5% de datos 
# Convertir columnas a formato fecha 

#Encontrar filas problemáticas
fechas_problematicas1 = ac[ac['fecha_registro'].notna() & pd.to_datetime(ac['fecha_registro'], errors='coerce').isna()]
# Mostrar diagnóstico completo
print(f"Se encontraron {len(fechas_problematicas1)} fechas problemáticas:")
print("Valores únicos problemáticos:")
print(fechas_problematicas1['fecha_registro'].unique())

# Mostrar filas completas para inspección
print("\nFilas completas con problemas:")
print(fechas_problematicas1)


#Transformar a formato fecha considerando estos formatos variables
def convertir_fecha(fecha_str):
    try:
        # Intenta con formato que incluye hora
        return pd.to_datetime(fecha_str, format='%Y-%m-%d %H:%M')
    except ValueError:
        try:
            # Intenta sin hora
            return pd.to_datetime(fecha_str, format='%Y-%m-%d')
        except ValueError:
            return pd.NaT  # Devuelve no-es-una-fecha para valores inválidos

ac['fecha_registro'] = ac['fecha_registro'].apply(convertir_fecha)
ac['fecha_registro'].dt.year.unique()

# Fecha from date 

# Encontrar filas problemáticas
fechas_problematicas2 = ac[ac['from_date'].notna() & pd.to_datetime(ac['from_date'], errors='coerce').isna()]
# Mostrar diagnóstico completo
print(f"Se encontraron {len(fechas_problematicas2)} fechas problemáticas:")
print("Valores únicos problemáticos:")
print(fechas_problematicas2['from_date'].unique())

#Finalmente, se combinan los dos filtros con & (AND lógico):

#(ac['from_date'].notna()): el valor existe (no es nulo)
#&
#(pd.to_datetime(ac['from_date'], errors='coerce').isna()): pero no es una fecha válida

ac['from_date'] = ac['from_date'].apply(convertir_fecha)
ac['from_date'].dt.year.unique()


# Fecha accidente
# Encontrar filas problemáticas
fechas_problematicas = ac[ac['fecha_acc'].notna() & pd.to_datetime(ac['fecha_acc'], errors='coerce').isna()]
#notna  devuelve True para los valores que no son nulos
# El parámetro errors='coerce' hace que si un valor no se puede convertir a fecha, lo transforme en NaT (Not a Time = valor nulo en fechas).
#Devuelve True para los valores que NO se pudieron convertir a fecha (es decir, los que quedaron como NaT).
# Mostrar diagnóstico completo
print(f"Se encontraron {len(fechas_problematicas)} fechas problemáticas:")
print("Valores únicos problemáticos:")
print(fechas_problematicas['fecha_acc'].unique())

# Mostrar filas completas para inspección
print("\nFilas completas con problemas:")
print(fechas_problematicas)

ac['fecha_acc'] = ac['fecha_acc'].apply(convertir_fecha)


ac["fecha_acc"].dt.year.unique()
# errors="coerce": Convierte los valores no válidos a NaT (Not a Time, equivalente a un NaN para fechas).
# Al revisar nuevamente los nulos son un 8% es poquito por lo que lo puedo imputar por la media

ac[ac["fecha_registro"].isnull()].head(20)

ac["fecha_registro"].isna().sum() 
print(ac["fecha_registro"].dtype) 

## Primer metodo a considerar imputar mean considerando diferencia entre fecha de registro y fecha acc
ac["diferencia_dias"] = (ac["fecha_registro"]-ac["fecha_acc"]).dt.days

ac["diferencia_dias"].describe()
# La desviación estándar (std) mide cuánto varían los datos con respecto a la media.
# El std es muy al 357.180, por lo que no es bueno imputar con ese metodo
#La dispersión (357 días) es 4.12 veces el valor promedio (86.68 días).
# Regla general para considerar baja o alta variabilidad:
#Baja variabilidad: std < 30% de la media
#Moderada variabilidad: 30% ≤ std ≤ 60% de la media
#Alta variabilidad: std > 60% de la media

# Segundo metodo los datos por interpolacion

# Graficar la serie de tiempo sin ordenar

import matplotlib.pyplot as plt

# Contar registros por día sin ordenar
registros_por_dia = ac["fecha_registro"].value_counts().sort_index()
registros_por_dia
plt.figure(figsize=(12, 5))
plt.plot(registros_por_dia.values, linestyle='-', linewidth=1)

# Personalizar gráfico
plt.title("Registros por Día (con Orden)")
plt.xlabel("Índice")
plt.ylabel("Cantidad de Registros")
plt.grid(True)

# Mostrar gráfico
plt.show()

# Usar Interpolación 📈 cuando:
# Los datos tienen una tendencia o patrón temporal claro.
# Se desea preservar la continuidad de la serie de tiempo.
# Los valores perdidos están rodeados de valores existentes.
# La diferencia entre valores consecutivos es relativamente pequeña.

# Usar la Media 📊 cuando:
# Los datos no tienen una tendencia clara y son más aleatorios.
# Hay una cantidad pequeña de datos faltantes y bien distribuidos.
# Se busca estabilidad sin introducir sesgos o distorsiones.
# No es necesario conservar la evolución natural de los datos.

# Tercer metodo usar la mediana, me quedare con ese.

# Supongamos que 'ac' es tu DataFrame y 'fecha_accidente' la columna con fechas
mediana_dias = 49  # Mediana de diferencia_dias

# Imputar los valores NaT en 'fecha_registro' sumando la mediana a 'fecha_accidente'
ac["fecha_registro"] = ac["fecha_registro"].fillna(ac["fecha_acc"] + pd.to_timedelta(mediana_dias, unit='D'))

# Verificar que no haya valores NaT restantes
print(ac["fecha_registro"].isna().sum())  # Debería ser 0


## Variable AMV 5% Nulos
ac["amv"].unique() 
ac["territorial"].unique() 


#Esto  mostrará todas las combinaciones únicas entre AMV y territorial, y cuántas veces aparecen.

relacion = ac.groupby("amv")["territorial"].size().reset_index(name='conteo')
print(relacion)

# size: Cuenta el número de elementos en cada grupo
#Para cada valor único de "amv", cuenta cuántos valores hay en "territorial"
# name='conteo' asigna el nombre "conteo" a la columna que contiene los resultados del conteo
#Los valores de "amv" que eran índices ahora se convierten en una columna normal

#Verificación de correspondencia 1 a 1

pd.options.display.max_rows = None 

# 1. Obtenemos AMVs con múltiples territoriales
amv_problematicos = ac.groupby("amv")["territorial"].nunique()
amv_problematicos = amv_problematicos[amv_problematicos > 1]

print("AMVs con múltiples territoriales asignados:")
print(amv_problematicos)

# 2. Para cada AMV problemático, mostramos sus territoriales
print("\nDesglose de territoriales por AMV problemático:")
for amv in amv_problematicos.index:
    territoriales = ac[ac["amv"] == amv]["territorial"].unique()
    print(f"AMV: {amv} → Territoriales asignados: {', '.join(territoriales)}")

agrupacion = ac.groupby("amv")["territorial"].unique()
agrupacion

# En este caso son solo tres acronimos los que no coinciden, por lo que se eliminara esa columna y se utilizada territorial.
# Ademas los acronimos no son vital para el estudio de la probabilidad de accidente

ac.drop(columns=["amv"], inplace=True)

## Variable n_victimas
ac["n_victimas"].unique()
ac["n_victimas"].describe()

import matplotlib.pyplot as plt
plt.boxplot(ac["n_victimas"].dropna())
plt.title("Distribución de Víctimas")
plt.show()

plt.figure(figsize=(10, 6))
plt.hist(ac['n_victimas'].dropna(), bins=20, color='skyblue', edgecolor='black')
plt.title('Distribución del Número de Víctimas por Accidente')
plt.xlabel('Número de Víctimas')
plt.ylabel('Frecuencia')
plt.grid(True, linestyle='--', alpha=0.7)
plt.show()

# Imputare por la mediana luego de ver un outsider y ver la distribucion asimetrica.

# Si 'ac' es un slice de otro DataFrame
ac = ac.copy()  # Creamos una copia explícita
mediana = ac["n_victimas"].median()
ac["n_victimas"] = ac["n_victimas"].fillna(mediana)
(ac.isnull().sum()/18553)*100

## Variable procedencia
ac["procedencia"].unique()
ac["procedencia"].describe()

# En este caso la variable procedencia no influye en la probabilidad de accidente
# PERO Sirve para analizar la calidad de los datos.

ac["procedencia"] = ac["procedencia"].fillna('DESCONOCIDO')

## Variable Revisado por 10% missing
#Analizando esta variable no es de interes para el estudio el nombre de quien evaluo no incide en los accidentes
ac["revisado_por"].unique()
ac.drop(columns=["revisado_por"], inplace=True)

## Variable To_date
#Aca puedo observar que to_date no tiene datos, todos son nulos
# Compruebo efectivamente y borro la variable de una
ac["to_date"].unique()

ac.drop(columns=["to_date"], inplace=True)

## Variable hora acc.

ac["fecha_acc"].dt.hour.unique()

ac["hora_acc"].unique()

# Crear tabla de comparación
comparacion = pd.crosstab(
    ac["fecha_acc"].dt.hour,
    ac["hora_acc"],
    dropna=False
)
print(comparacion)

# Verificar si la distribución horaria varía según la procedencia
if "procedencia" in ac.columns:
    print(
        ac.groupby('procedencia')['fecha_acc']
        .apply(lambda x: x.dt.hour.value_counts())
        .unstack()
    )
# Como tal pareciera que la hora en fecha_acc fuese algo automatico, por lo que me quedare con hora_acc
# Convertir a entero (si es float)
ac["hora_acc"].describe()

import matplotlib.pyplot as plt

plt.figure(figsize=(10, 6))
plt.hist(ac["hora_acc"], bins=24, edgecolor='black')
plt.title('Distribución de Accidentes por Hora del Día')
plt.xlabel('Hora')
plt.ylabel('Número de Accidentes')
plt.xticks(range(0, 24))
plt.grid(axis='y', alpha=0.3)
plt.show()

# En la grafica se ve que tiene dos picos a las 7 ama como a las 5 pm
#Transformar de float a int para poder utilizar la distribucion como imputacion 

# Convertir valores válidos a enteros
ac['hora_acc'] = ac['hora_acc'].dropna().astype(int)
# dropna no modifica de forma permamnetne la columna, sino temporal.

# Verificar valores únicos
print(sorted(ac['hora_acc'].unique()))

import numpy as np
# 1. Calcular distribución de frecuencia
frecuencia = ac['hora_acc'].value_counts(normalize=True).sort_index()

#value_counts(normalize=True) cuenta cuántas veces aparece cada valor y lo convierte a proporciones (es decir, suma 1).

# 2. Imputar los NaN con base en la distribución
ac['hora_acc'] = ac['hora_acc'].apply(
    lambda x: np.random.choice(frecuencia.index, p=frecuencia.values) if pd.isna(x) else int(x))

#Aplica una función lambda a cada fila de la columna hora_acc.

#pd.isna(x): Verifica si el valor es nulo (NaN).
#Si es nulo, selecciona un valor aleatorio de los índices (frecuencia.index) siguiendo la distribución de probabilidades (frecuencia.values) usando np.random.choice().
#Si NO es nulo, simplemente lo convierte a entero (int(x)) porque a veces puede venir como float.


## Variable Heridos 

ac["n_heridos 1"].unique()
ac["n_heridos"].unique()
# Hay otra variable heridos y completa
#Borrare por tanto la varibale "n_heridos 1"

ac.drop(columns=["n_heridos 1"], inplace=True)

## Variable Fuente
ac["fuente"].unique()
# No es una variable reelevante para el estudio solo dice desktop, se borrara.
ac.drop(columns=["fuente"], inplace=True)

# Características de del
# Modifica el DataFrame directamente (no necesitas inplace=True).
# Es más rápido que drop(), pero no puedes eliminar filas con del.
# No devuelve nada, solo elimina la columna de inmediato.

## Variable Observa

ac["observa"].unique()

#array([nan,
#       'Se envío a planta central y a la DT-CAS informe de accidentes y registro fotográfico',
#       'En el informe policial registran el sitio del accidente en el PR13+0700, de acuerdo con el sistema de referenciación de INVIAS el PR correcto es 13+0620.',
#       ..., 'Sin observaciones.', 'No hay observación.',
#       'No se tiene información de los vehículos involucrados y sus sentidos.'],
#      dtype=object)
# No es una variable necesaria por:
# Si los comentarios son opcionales y no están en todos los registros, su ausencia puede hacer que la variable no sea consistente
# Sin observacion o datos como no hay observacion no suman al analisis

ac.drop(columns=["observa"], inplace=True)

## Variable causa_old ##

# Ahora revisaremos un cruce haber si con el match de causa-old puedo saber cuales son esas categorias con numeros.
ac.groupby("Causa_old")["causa_posible"].unique()
ac.groupby("causa_posible")["Causa_old"].unique()

# Se encuenta que las categorias 0 al 10 tienen nombres que se corresponden con causa_old
# Revisaremos con rigor si se perdieron datos al pasar de causa_old a causa_posible parece un tipo de normalizacion.
# Contar valores únicos en ambas columnas
# Valores únicos en cada columna
causa_old_unicas = set(ac["Causa_old"].explode().dropna().unique())
causa_old_unicas
causa_posible_unicas = set(ac["causa_posible"].explode().dropna().unique())
causa_posible_unicas
# explode()  Divide listas en múltiples filas individuales.
# Diferencias: lo que está en "Causa_old" pero NO en "causa_posible"

# Elimina valores nulos o NaN.
# Después de explotar, algunas filas pueden contener "nan" o valores vacíos.
# Esto elimina esas filas.
perdidos = causa_old_unicas - causa_posible_unicas  # Causas que estaban antes pero ya no están
nuevos = causa_posible_unicas - causa_old_unicas    # Causas nuevas que no estaban en el original

print("Causas perdidas tras la normalización:", perdidos)
print("Causas nuevas tras la normalización:", nuevos)

##¿Por qué esto mejora la revisión?
##Permite ver exactamente qué causas desaparecieron.
##No solo agrupa, sino que compara directamente ambas listas.
##Si causas_perdidas no está vacío, significa que hubo datos eliminados o fusionados incorrectamente.


# Revisar ejemplos de correspondencia
muestra = ac[["Causa_old", "causa_posible"]].drop_duplicates().head(20)
print(muestra)

# Revisar una sola categoria

muestra = ac[ac["causa_posible"] == "Impericia en el manejo"][["Causa_old", "causa_posible"]].drop_duplicates().head(20)
print(muestra)

# He decido trabajar sobre la columna Causa_posible debido a que no tiene missing value.
# Aunque muchas diferencias, es la variable que finalmente transformaron con base a los datos y sustento del registro.
#isna() y isnull():son lo mismo, la form mas moderna es isna()
muestra_nan = ac[ac["Causa_old"].isna()][["Causa_old", "causa_posible"]].sample(20, random_state=42)
print(muestra_nan)
# sample(20, random_state=42) → Toma una muestra aleatoria de 20 filas para revisar.
#random_state=42 → Asegura que la muestra sea reproducible si vuelves a ejecutar el código.

ac.drop(columns=["Causa_old"], inplace=True)

## Me quedare con la variable causa_posible, asumiendo que la normalizacion fue hecha coon base a informes adicionales que les permitio en algunas casos redefinir la categorizacion.
## Ademas le colocare nombre a las categorias con numeros con base a la variable causa_old. Trabajare sin tildes

ac["causa_posible"] = ac["causa_posible"].replace("0", "No determinada")
ac["causa_posible"] = ac["causa_posible"].replace("1", "Exceso de velocidad")
ac["causa_posible"] = ac["causa_posible"].replace("2", "Fallas mecanicas")
ac["causa_posible"] = ac["causa_posible"].replace("3", "Embriaguez del conductor")
ac["causa_posible"] = ac["causa_posible"].replace("4", "Daños de la calzada")
ac["causa_posible"] = ac["causa_posible"].replace("5", "Vehiculo, objeto, persona o animal en la via")
ac["causa_posible"] = ac["causa_posible"].replace("6", "Imprudencia del peaton")
ac["causa_posible"] = ac["causa_posible"].replace("7", "Imprudencia del conductor")
ac["causa_posible"] = ac["causa_posible"].replace("8", "Deslizamiento o derrumbe")
ac["causa_posible"] = ac["causa_posible"].replace("9", "Orden publico]")
ac["causa_posible"] = ac["causa_posible"].replace("10", "Caida de piedra / Alud superior")

ac["causa_posible"].unique()

## Vamos a corregir este nombre 
#  'Adelantar invadiendo caril de sentido contrario'
ac["causa_posible"] = ac["causa_posible"].str.replace("caril", "carril", regex=False)
ac["causa_posible"] = ac["causa_posible"].str.replace("pasarjero", "pasajero", regex=False)
ac["causa_posible"] = ac["causa_posible"].str.replace("orden publico]", "orden publico", regex=False)
# regex=False: Indica que no se está usando una expresión regular, solo un reemplazo directo.

## Agrupar por categorias similares

# Antes dejar todo sin tildes y minusculas para evitar duplicados
# Instalar en la terminal pip install Unidecode
from unidecode import unidecode

ac["causa_posible"] = ac["causa_posible"].str.lower().apply(unidecode).str.strip()
ac["causa_posible"]
#.str.lower() → convierte todo a minúsculas.
#.apply(unidecode) → elimina tildes y convierte letras especiales a ASCII puro (á → a, ñ → n, etc.).
#.str.strip() → elimina espacios al principio y final.


# Categoría: Imprudencias del conductor (excluyendo "Impericia en el manejo")
# Se realizaran subcategorias

ac["causa_posible"] = ac["causa_posible"].replace({
    "embriaguez del conductor": "embriaguez o sustancias alucinogenas",
    "embriaguez aparente": "embriaguez o sustancias alucinogenas",
    "cruzar en estado de embriaguez": "embriaguez o sustancias alucinogenas"
})

#El método np.where() de NumPy funciona como un "if-else" vectorizado, para trabajar en grandes
#conjuntos de datos.
import numpy as np
ac["causa_posible"] = np.where(ac["causa_posible"].isin([
    "adelantar cerrando",
    "adelantar por la derecha",
    "adelantar en curva o en pendientes",
    "adelantar en zona prohibida",
    "adelantar invadiendo carril de sentido contrario",
    "adelantar invadiendo carril del mismo sentido en zigzag",
]), "adelantamientos indebidos", ac["causa_posible"])

ac["causa_posible"] = np.where(ac["causa_posible"].isin([
    "estacionar sin seguridad",
    "falta de senales en vehiculo varado",
    "vehiculo mal estacionado",
    "no hacer uso de senales reflectivas o luminosas",
    "aprovisionamiento indebido",
    "dejar obstaculos en la via",
    "reparar vehiculo en via propia",
    "carga sobresaliente sin senales",
    "carga sobresaliente sin autorizacion"
]), "estacionamiento y senalizacion imprudente", ac["causa_posible"])

# Transportar personas u objetos de forma inadecuada dentro o sobre un vehículo, lo cual puede generar riesgos, son maniabros indebidas"
#Circular por el extremo derecho, incluso fuera de la calzada, en vías no pavimentadas, lo que podría poner en riesgo al conductor y a otros (por baches, lodo, animales o falta de visibilidad
ac["causa_posible"] = np.where(ac["causa_posible"].isin([
    'realizar giro en "u"',
    "reverso imprudente",
    "poner en marcha un vehiculo sin precauciones",
    "arrancar sin precaucion",
    "frenar bruscamente",
    "remolque sin precaucion",
    "girar bruscamente",
    "salirse de la calzada",
    "transitar en contra via",
    "subirse al anden o vias peatonales",
    "transitar uno al lado del otro",
    "transitar por su derecha en vias rurales"
]), "maniobras imprudentes", ac["causa_posible"])

ac["causa_posible"] = np.where(ac["causa_posible"].isin([
    "desobedecer senales o normas de transito",
    "desobedecer el agente",
    "no respetar la prelacion",
    "no esperar prelacion de intersecciones o giros",
    "transitar en contravia",
    "transitar sin luces",
    "transitar sin los dispositivos luminosos de detencion",
    "no cambiar luces",
    "semaforo en rojo",
    "transitar por vias prohibidas",
    "transportar pasajeros en vehiculos de carga",
    "transitar otra persona o cosas"
]), "desobediencia a normas y senales", ac["causa_posible"])

# "exceso en horas de conduccion"
# "exceso de velocidad "
# "imprudencia del conductor"

ac["causa_posible"] = np.where(ac["causa_posible"].isin([
    "exceso de peso",
    "transporte de carga sin seguridad"
]), "imprudencia del conductor", ac["causa_posible"])
# "No mantener ditancia de seguridad"
# "defectos fisicos y psiquicos"
# "otra conductor en general" La causa del incidente o infracción está relacionada con el comportamiento del conductor,

ac["causa_posible"] = np.where(ac["causa_posible"].isin([
    "fallas en los frenos",
    "fallas en la direccion",
    "fallas en el sistema electrico",
    "fallas en luces delanteras",
    "fallas en luces direccionales",
    "fallas en luces posteriores",
    "fallas en luces de frenos",
    "fallas mecanicas",
    "falla en el limpia brisas",
    "fallas en ajuste capo",
    "fallas en el tubo de escape",
    "falta de mantenimiento mecanico",
    "fallas en el sistema electrico",
    "fallas en las llantas",
    "fallas en las puertas ",
    "fallas en el limpia brisas",
    "fallas en el tubo de escape. gases en el interior del vehiculo",
    "incendio por reparacion indebida",
    "otra - del vehiculo",
    "fallas en las puertas",
    "conducir vehiculo sin adaptaciones" 
]), "fallas mecanicas y del sistema electrico", ac["causa_posible"])


ac["causa_posible"] = np.where(ac["causa_posible"].isin([
    "ausencia total o parcial de senales",
    "danos de la calzada",
    "deslizamiento o derrumbe",
    "superficie humeda",
    "superficie lisa",
    "huecos",
    "falta de precaucion por niebla, lluvia o humo",
    "obstaculos en la via",
    "dejar o movilizar semovientes en la via",
    "otras - de la via",
    "vehiculo, objeto, persona o animal en la via",
    "caida de piedra / alud superior",
    "ausencia o deficiencia en demarcacion",
    "falta de prevencion ante animales en la via"
]), "condiciones de la via y factores ambientales", ac["causa_posible"])

# Clasificación de conducta de peatones
ac["causa_posible"] = np.where(ac["causa_posible"].isin([
    "imprudencia del peaton",
    "pararse sobre la calzada",
    "otras -del peaton",
    "cruzar sin observar",
    "jugar en la via",
    "transitar por la calzada",
    "pararse sobre la clazada",
    "transitar distante de la acera u orilla de la calzada",
    "transitar entre vehiculos",
]), "conducta de peatones", ac["causa_posible"])

# Clasificación de conducta de pasajeros
ac["causa_posible"] = np.where(ac["causa_posible"].isin([
    "pasajeros obstruyendo el conductor o sobrecupo",
    "transportar pasajeros en la parte exterior",
    "recoger o dejar pasajeros sobre la calzada",
    "dejar o recoger pasajeros en sitios no demarcados",
    "salir por delante de un vehiculo",
    "descender o subir del vehiculo en marcha",
    "viajar colgado o en los estribos",
    "pasajero embriagado"
]), "conducta de pasajeros", ac["causa_posible"])
# otra-pasajero o acompanante


ac["causa_posible"].value_counts()

###############################
ac.isnull().sum()

#### 6. IDENTIFICACION DE VARIABLES CATEGORICAS

ac.info()
ac.head()

# astype() es un método de los objetos de pandas como Series y 
# DataFrame que se usa para cambiar el tipo de dato (dtype) de una columna o de todo el DataFrame.

cat_cols = ac.select_dtypes(include='object').columns
ac[cat_cols] = ac[cat_cols].astype('category')

#### 7. DEFINIR VARIABLES CON LAS CUALES REALIZARE MI OBJETIVO DE ESTUDIO

# Mi objetivo es conocer la probabilidad de un accidente con base a una serie de condiciones, que a la vez induce al analisis de las causas
# Análisis predictivo con enfoque explicativo
# Predecir la ocurrencia de un accidente y a la vez entender las causas asociadas.


## Vamos a ver que significa cada uno y sus categorias unicas

#- objectid: Es el identificador. Codigo Es otro identificador y eventid
# Podemos prescindir de ellos por que son identificadores sin valor predictivo.

ac.drop(columns=["objectid"], inplace=True)
ac.drop(columns=["codigo"], inplace=True)
ac.drop(columns=["eventid"], inplace=True)


#fecha_acc	Te sirve para crear variables como mes, año, hora, etc.
#dia_semana_acc	Día puede influir en el riesgo (lunes ≠ domingo).
#min_acc, hora_acc	Tiempo exacto del evento.
#condic_meteor	Muy relevante: clima afecta siniestralidad.
#estado_super	Estado de la superficie de la vía.
#terreno	Pendiente o relieve también influye.
#secc_tip	Tipo de sección vial (curva, recta...).
#tipo_cierre, horas_cierre, min_cierre	Indica si había cierre en la vía.
#clase_accidente	Útil para análisis complementario o modelar tipo.
#causa_posible	Puedes analizar causas más frecuentes.

ac["lado"].unique()
# No encuentro como tal en la metadato un significado por lo que se eliminara
ac.drop(columns=["lado"], inplace=True)
# PR indica el numero del PR Indica el número de kilómetro de la vía donde se encuentra.
ac["pr"].unique()
# podria dejarse para una analisis detallado de en que vias usalmente ocurre un accidente
# ac["km_aprox"] = ac["pr"] + ac["distancia_pr"]  ubicación más precisa
# Por lo que la variable distancia_pr se queda.
#Podrías hacer un gráfico de densidad o un histograma de km_aprox por vía, para ver zonas de concentración de accidentes.


# Se continua investigando las variables relacionadas con la ubicacion de al via
# ref_loc 0101_0_22	 Codigo Tramo	0101 Poste de Referencia Inicial	0Poste de Referencia Final	22
#https://inviasopendata-invias.opendata.arcgis.com/datasets/016b917815cd49eb9a9bcdbe88a5c8ca/about

# Se procede a borrar la variable codigo de la via por que ya esta en ref_loc, mas no se borra pr y distancia por lo mencionado anteriormente, sirve para ubicacion precisa

ac["codigo_via"].unique()

ac.drop(columns=["codigo_via"], inplace=True)

# Se procede a borrar la variable ruta_id por que es un subconjunto de la variabla ref_loc

ac.drop(columns=["ruta_id"], inplace=True)

# Variable procedencia se elimina pues nada aporta saber si fue suministrada por desconocidos, por policia o testigos
ac["procedencia"].unique()
ac.drop(columns=["procedencia"], inplace=True)

# Variable ref_met no contiene informacion relevante en todos se repite post ref y ya, se elimina
ac["ref_met"].value_counts()

ac.drop(columns=["ref_met"], inplace=True)

# Revisemos la variable LocError 
ac["LocError"].value_counts()
"""LocError
NO ERROR                    18548
ROUTE NOT FOUND                 3
ROUTE LOCATION NOT FOUND        2
"""
# Filtrar todas las filas que contienen "ROUTE LOCATION NOT FOUND"
error_ruta = ac[ac["LocError"].str.contains("ROUTE LOCATION NOT FOUND", na=False)]
error_ruta
error_ruta1 = ac[ac["LocError"].str.contains("ROUTE NOT FOUND", na=False)]
error_ruta1
#Al revisar las filas que contiene, ruta no localizada vemos que si estan en la variable ref_loc

ac.drop(columns=["LocError"], inplace=True)

#Vamos a mirar la variable distancia_pr y ref_off 

(ac["ref_off"] == ac["distancia_pr"]).value_counts()
# True    18553
# Es decir que tienen los mismos valores puedo borrar una como ref_off

ac.drop(columns=["ref_off"], inplace=True)


#Vamos con la variable from_date 
#Cuando yo la veo a simple vista no es logica por que los accidentes tiene una fecha año mayor a from date

ac['dias_despues_acc'] = (ac['from_date'] - ac['fecha_acc']).dt.days
ac['dias_despues_acc']

# No tiene sentido ni como registro, por que es previo al registro, a lo mejor es algun error en la data,
# Asi que se eliminara
ac.drop(columns=["from_date"], inplace=True)


## Quedamos pendiente de la variable meas

# No se encontro alguna logica de la base por lo que se elimino
ac.drop(columns=["meas"],inplace=True)

# Ademas se borraron las categorias diferencia_dias y dias_despues que creamos para imputar datos
ac.drop(columns=["diferencia_dias"],inplace=True)
ac.drop(columns=["dias_despues_acc"],inplace=True)

# Se organizo la variable categorica  dia semana acc
ac["dia_semana_acc"] = ac["dia_semana_acc"].str.lower().apply(unidecode).str.strip()


####ANALISIS UNIVARIADO(por variable individual)

###Variables Numericas

##Resumen estadístico

ac.describe()

# Tener presente que pr y distancia-pr es mas para localizacion geografica exacta del accidente
# Porlo que pr y distancia pr no se los considerara las estadisiticas.
# No obstante, si nos vamos a las horas del accidente y minuto del accidente.

# Hora Accidente Y minuto Accidente

# Podemos analizar el tiempo en que mas accidentes ocurre, en promedio sobre el medio dia al minuto 21, ocurren los accidentes
# Al observar la desviacion standard que es la distancia de los datos frente a la media
# Encontramos que es de 5.92, por lo que al mirar desde el coeficiente 5.92/12.29 = 48.17%  tiene una
# variabilida moderada.
# Adicional se puede interpretar como 12+- 5.92  Entre 6.1 y 17.9 horas ocurren la mayoria de acciones
# Si fuera normal tenemos esto 
# En cuanto a los minutos  17.38/21.877= 80% m laa variabilidad es muy alta.
# 21.877 +- 17.38 entre entre 4.497 39.25

#| Rango   | Porcentaje de datos en una distribución normal |
#| --------| ---------------------------------------------- |
#|`μ ± 1σ` | ≈ 68% de los datos                             |
#|`μ ± 2σ` | ≈ 95%                                          |
#|`μ ± 3σ` | ≈ 99.7%                                        |

# Histograma + Kde Hora Accidente
#KDE significa Kernel Density Estimation

#Es una técnica no paramétrica para estimar la función de densidad de probabilidad de una variable aleatoria continua.
#Muy útil para visualizar la distribución de datos, especialmente cuando no se quiere asumir que siguen una distribución específica como la normal.
# La PDF nos dice qué tan probable es que un valor esté cerca de cierto punto, aunque no da probabilidades exactas para un valor puntual.
# Lo ideal es que ese rango cubra tus datos, con un pequeño margen a los lados.

# pip install pandas matplotlib seaborn antes de

import seaborn as sns
import matplotlib.pyplot as plt

# Crear histograma + KDE bins de 24 horas
plt.figure(figsize=(10, 6))
sns.histplot(ac["hora_acc"], kde=True, bins=24, color='skyblue', edgecolor='black')

# Etiquetas y título
plt.title("Histograma + KDE de hora_ac")
plt.xlabel("Hora")
plt.ylabel("Frecuencia / Densidad")

# Mostrar gráfico
plt.grid(True)
plt.tight_layout()
plt.show()

#Conclusion: NO sigue una distribución normal.
# Tiene dos modos o picos, por eso la distribución es bimodal.
# pico de la mañana (inicio de jornada laboral) y pico de la tarde (fin de jornada / congestión de regreso a casa).

# Variable n victimas

# El promedio fue de 1.11 y la std 1.97 1.97/1.11. Se ven muy disperso los datos en 171 % segun el coeficiente.
# A lo mejor se deben a outsider veo un valor maximo de 123 que discrepa con el promedio y valor minimo

# Usare el metodo cuartiles (IQR) para verificar que es un outlier

sns.boxplot(x=ac['n_victimas'])
plt.show()

#Se eliminó el valor máximo (123) en la variable accidente por considerarse un valor atípico extremo que distorsiona la media, no representativo del comportamiento general de los datos.
ac_filtrado = ac[ac['n_victimas'] != ac['n_victimas'].max()]
ac_filtrado
ac_filtrado.describe()


# Ahora trabajaremos con ac_filtrado

#Promedio 1.10 y desviacion de 1.75, la dispercion sigue siendo algo, aunque bajo.
# En general en promedio hay una victima por accidente

# Crear histograma + KDE bins de 24 horas
plt.figure(figsize=(10, 6))
sns.histplot(ac_filtrado['n_victimas'], kde=True, bins=30, color='skyblue', edgecolor='black')

# Etiquetas y título
plt.title("Histograma + KDE de n_victimas")
plt.xlabel("n_victimas")
plt.ylabel("Frecuencia / Densidad")

# Mostrar gráfico
plt.grid(True)
plt.tight_layout()
plt.show()

#Conclusion La gran mayoría de los valores están entre 0 y 2 víctimas
#Distribucion sesgada positiva o asimétrica a la izquierda
#Probablemente sigue una distribución de Poisson o binomial negativa, típica de conteos de sucesos poco frecuentes

# Variable numero de muertos
# Al revisar el promedio de muertes es menos de 1, 0.10 y el maximo son 9. Std 0.37.Lo que indica que, en la mayoría de los accidentes registrados, no se reportan fallecidos.
# 0.37/0.10 una variabilidad alta

#n_victimas	Persona(s) reconocida(s) oficialmente como víctima(s) — posiblemente según criterios legales o administrativos.
#muertos	Personas fallecidas.
#heridos	Personas lesionadas, sin importar su estatus como "víctima legal".
ac_filtrado.n_victimas.unique()

# Aclaramos conceptos por que suena muy similar en el analisis

# Variable numero de heridos
# El promedio es de 0.59, osea que en promedio no hay personas heridas en el accidente.
# Aunque a lo mejor si se presentan accidentes donde puede haber una mayor cantidad de heridas hasta 35
# La desviacion standard es de 1.28 y el coef 1.28/0.59=2.16 siempre hay una variabilidad 
# Esta variabilidad se debe a que pueden haber uno que otro registro con una mayor cantidad de heridos de lo normal.
# Si observamos la mediana de heridos es 0.

import matplotlib.pyplot as plt
import seaborn as sns

# Histograma + KDE para la variable n_heridos
plt.figure(figsize=(10, 6))
sns.histplot(ac["n_heridos"], kde=True, bins=range(0, ac["n_heridos"].max() + 2), color='salmon', edgecolor='black')

# Etiquetas y título
plt.title("Histograma + KDE de número de heridos")
plt.xlabel("Número de heridos")
plt.ylabel("Frecuencia / Densidad")

# Mostrar gráfico
plt.grid(True)
plt.tight_layout()
plt.show()


#La gran mayoría de los accidentes tienen 0 a 2 heridos.

#Muy pocos casos presentan más de 5 heridos.

#Casos extremos (como 20, 30 heridos) son raros, pero existen.

# Variable horas_cierre y min_cierre

# En promedio las horas de cierra en la via son 5 con 12 minutos
# La desviacion es de 7h, lo cual me dice que hay una diferencia de 2,  es mucha la variabilidad
# coef varia 7/5 un 140%
# Asi mismo la diferencia en minutos es 4 de minutos en cuanto desviacion std es bastante la variabilidad


# Crear histograma + KDE bins de 24 horas
plt.figure(figsize=(10, 6))
sns.histplot(ac["horas_cierre"], kde=True, bins=24, color='skyblue', edgecolor='black')

# Etiquetas y título
plt.title("Histograma + KDE de horas_cierre")
plt.xlabel("Hora")
plt.ylabel("Frecuencia / Densidad")

# Mostrar gráfico
plt.grid(True)
plt.tight_layout()
plt.show()

ac_filtrado.head()
ac_filtrado.clase_accidente.unique()
# Este seria un dato que indica a que horas se reestablece la via usalmente entre alas 0 y 5 horas, dependiendo del accidente.

def calcular_duracion(hora_acc, hora_cierre):
    if hora_cierre >= hora_acc:
        return hora_cierre - hora_acc
    else:
        return (24 - hora_acc) + hora_cierre  # cierre al día siguiente

ac_filtrado['duracion_cierre_horas'] = ac_filtrado.apply(lambda row: calcular_duracion(row['hora_acc'], row['horas_cierre']), axis=1)


## Variables categoricas analisis

ac_filtrado.info()

#Consiste en:
#Contar cuántas veces aparece cada categoría.
#Calcular proporciones (%).
#Visualizar con gráficos de barras o pastel (opcional).

# col es una variable temporal
variables_cat = [
    'territorial', 'dia_semana_acc', 'condic_meteor', 'estado_super', 'terreno',
    'secc_tip', 'geometria_acc', 'tipo_cierre', 'clase_accidente',
    'causa_posible']

for col in variables_cat:
    print(f"\n🔹 Variable: {col}")
    conteo = ac_filtrado[col].value_counts()
    porcentaje = ac_filtrado[col].value_counts(normalize=True) * 100

    resumen = pd.DataFrame({'Frecuencia': conteo, 'Porcentaje (%)': porcentaje.round(2)})
    print(resumen)

    # Gráfico
    plt.figure(figsize=(10, 5))
    sns.countplot(data=ac_filtrado, y=col, order=conteo.index, palette='pastel')
    plt.title(f"Distribución de {col}")
    plt.xlabel("Frecuencia")
    plt.ylabel(col)
    plt.tight_layout()
    plt.show()

#Conclusiones
# Los territorios donde occurren mas accidentes en Planta Central(sede principal en Bogotá), Antioquia y Valle del cauca.
# Estos tres territorios ocupan el 30% de los accidentes
# La mayoria de accidentes ocurre apartir del jueves y hasta el domingo, con un acumulado de 37%. El dia de mas accidente el domingo
# El dia de menos accidentes es el martes
# Las condiciones metereologicas al momento del accidente eran normales un 65.55% de los casos. En segundo lugar fue durante la lluvia 15.40%
# Lo anterior nos dice que a lo mejor no fue lo que incidio en los accidentes.
# Ademas la mayoria de los accidentes ocurre en terreno plano 47.55%
# E incluso fue en rectas con un porcentaje del 60.78%
# El 65.39 ocurrio durante un transito normal con precaucion
# La mayoria de accidente se deben a choques con un 74.91%
# La mayoria de accidente no tuvo una cauda determinada, no obstante el no mantener distancia y las maniobras imprudentes 
# representaron el 22.54% de los causales

###  Análisis bivariado 

#Tipo de variables	¿Qué hacemos?	¿Para qué sirve?
#Numérica vs Numérica	Correlación, scatterplot	Ver si se mueven juntas
#Categórica vs Numérica	Boxplot, medias por grupo, ANOVA	Ver si cambia la media entre categorías
#Categórica vs Categórica	Tablas de frecuencia, chi-cuadrado	Ver si se asocian o no

#### 1. Día de la semana vs número de heridos
# BoxPlot
import seaborn as sns
# Importa la librería Seaborn, que se usa para hacer gráficos estadísticos más bonitos
import matplotlib.pyplot as plt
# Importa Matplotlib, la librería más común para hacer gráficos en Python.
#Seaborn trabaja encima de Matplotlib.
# Lista con el orden correcto de los días
orden_dias = ['lunes', 'martes', 'miercoles', 'jueves', 'viernes', 'sabado', 'domingo']

plt.figure(figsize=(10, 5))
sns.boxplot(x='dia_semana_acc', y='n_heridos', data=ac_filtrado,order=orden_dias)

plt.title('Heridos por día de la semana')
plt.xlabel('Día de la semana')
plt.ylabel('Número de heridos')
plt.xticks(rotation=45)
# Rota las etiquetas del eje X 45 grados, para que no se encimen y se lean mejor.
plt.tight_layout()
#Ajusta automáticamente los márgenes y espacios del gráfico para que todo quede bien acomodado y no se corte.
plt.show()


#    |-----[■■■]-----|
#    2     4 5 7    10

#Un boxplot resume
# Mínimo	El valor más bajo 2
# Q1 25% de los datos son menores
# Mediana (Q2)	El valor del medio
# Q3 75% de los datos son menores
# Máximo	El valor más alto 10

#Resultados
#La caja azul represente el 50% la mita de los accidente est entre  0  y 1
# Como la caja es muy pegada al 0, significa que casi todos los accidentes tienen pocos heridos.
# Los bigotes indican que llegan hasta 3 heridos.
# Puntos sueltos Accidentes poco comunes con muchos heridos

# Calculo de media por grupo

media_heridos = ac_filtrado.groupby("dia_semana_acc")["n_heridos"].mean().reindex([
    "lunes", "martes", "miercoles", "jueves", "viernes", "sabado", "domingo"
])
print(media_heridos)
#En general en promedio no suelen haber heridos
# Los días con más promedio de heridos (como sábado o domingo) podrían indicar accidentes más graves o más comunes en esos días.
# Graficar la media por día
media_heridos.plot(kind='bar', color='skyblue', edgecolor='black')
plt.title("Promedio de heridos por día de la semana")
plt.xlabel("Día de la semana")
plt.ylabel("Promedio de heridos")
plt.xticks(rotation=45)
plt.grid(axis='y')
plt.tight_layout()
plt.show()

#ANOVA (Análisis de Varianza) compara medias entre 3 o más grupos para saber si al menos uno es diferente.
#Analysis Of VAriance
#El ANOVA analiza cuánta variación hay entre los grupos (por ejemplo, entre los días de la semana) comparada con la variación dentro de los grupos (dentro de cada día).
#“¿El número promedio de heridos cambia significativamente según el día de la semana?”

import scipy.stats as stats

# Crear listas de heridos por cada día
grupos = [group["n_heridos"].values for name, group in ac_filtrado.groupby("dia_semana_acc")]

# Una lista por comprensión (en inglés list comprehension) es una forma corta de crear listas basadas en un bucle (for), todo en una sola línea.
# Hazme una lista con los resultados de hacer algo con cada elemento de un conjunto.
# ac.groupby("dia_semana_acc")=Agrupa tu base de datos ac por el campo día de la semana, entrega 7 grupos.
# [expresión for elemento in colección]

#for name, group in ac.groupby("dia_semana_acc")	Recorre cada grupo de días de la semana
#group["n_heridos"].values	Saca los valores de la columna n_heridos solo de ese día
#“Para cada grupo que se forma al agrupar por dia_semana_acc, guárdame:
#El nombre del grupo en name,
#El subconjunto de datos (el grupo en sí) en group.”

f_stat, p_value = stats.f_oneway(*grupos)

print("Estadístico F:", f_stat)
print("Valor p:", p_value)

#p < 0.05	✅ Hay diferencias significativas entre al menos dos días
#p >= 0.05	❌ No hay evidencia estadística de diferencias en el promedio de heridos

#El estadístico F: mide cuánta diferencia hay entre los grupos
#Aunque los lunes, sábados o domingos puedan tener más o menos heridos en algunos casos, esa diferencia puede ser solo por azar.
#Estadísticamente, los promedios se parecen mucho entre los días.

#### 2.Analisis ¿La duración del cierre (en horas) cambia según el tipo de accidente?

#Así ves si hay clases con pocos datos que podrían no ser representativas.

ac_filtrado["clase_accidente"].value_counts()


grupos2 = [grupo["duracion_cierre_horas"].values for nombre, grupo in ac_filtrado.groupby("clase_accidente")]

f_stat, p_value = stats.f_oneway(*grupos2)

print("Estadístico F:", f_stat)
print("Valor p:", p_value)

#No hay evidencia suficiente para afirmar que la duración del cierre varíe significativamente entre los diferentes tipos de accidente.
## Miremos un boxplot haber que mas se puede encontrar

plt.figure(figsize=(12, 6))
sns.boxplot(x="clase_accidente", y="duracion_cierre_horas", data=ac_filtrado)
plt.xticks(rotation=45)
plt.title("Duración del cierre según tipo de accidente")
plt.xlabel("Clase de accidente")
plt.ylabel("Duración del cierre (horas)")
plt.grid(True)
plt.tight_layout()
plt.show()
#Las medianas (línea dentro de cada caja) son bastante similares en la mayoría de clases con cuerda con el ANOVA

### 3. Tablas de contigencia con la variable clase_accidente
variables = ['condic_meteor', 'estado_super', 'terreno', 'tipo_cierre', 'causa_posible']

from scipy.stats import chi2_contingency

for var in variables:
    print(f"\n📊 Variable: {var}")
    
    # Crear tabla de contingencia
    tabla = pd.crosstab(ac_filtrado[var],ac_filtrado['clase_accidente'])
    print(tabla)
    
    # Prueba chi-cuadrado
    chi2, p, dof, expected = chi2_contingency(tabla)
    print(f"Chi-cuadrado = {chi2:.2f}, p-valor = {p:.4f}, DOF = {dof}")
    
    if p < 0.05:
        print("➡️ Hay relación significativa con clase_accidente")
    else:
        print("⛔ No hay relación significativa")
       
# La variable clase accidente tiene relacion significativa con
# Condiciones metereologicas: En condiciones de lluvia, el número de choques y volcamientos es mucho mayor que en otras condiciones.
#en condiciones de sol o clima seco, predominan también los choques, pero con menor número de volcamientos.
#Lluvia y nieve aumentan riesgo de salidas de vía y volcamientos.
# Estado del suelo En superficies húmedas, hay un número alto de choques y volcamientos.
#Superficie seca y limpia también tiene muchos accidentes, pero con una distribución más variada.
# Terreno Terrenos montañosos y escarpados tienen más volcamientos y salidas de vía.
#Terreno plano concentra más choques y atropellos.
# Tipo de cierre Cuando hay tránsito normal con precaución, se producen más choques, atropellos y volcamientos.
#Los cierres parciales o totales también concentran volcamientos, pero en menor medida.
# Asi como con causa accidente
#No mantener distancia y maniobras imprudentes → muchos choques.
#Condiciones ambientales y fallas mecánicas → más volcamientos.
#Conducta del peatón → relacionado con atropellos.
#Exceso de velocidad → altamente vinculado con salidas de vía y vuelcos.

for var in variables:
    tabla = pd.crosstab(ac_filtrado[var],ac_filtrado['clase_accidente'])

    plt.figure(figsize=(10, 6))
    sns.heatmap(tabla, annot=True, fmt="d", cmap="YlGnBu")
    plt.title(f'Heatmap: clase_accidente vs {var}')
    plt.xlabel(var)
    plt.ylabel('Clase de Accidente')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()


for var in variables:
    tabla = pd.crosstab(ac_filtrado[var],ac_filtrado['clase_accidente'])
    tabla_norm = tabla.div(tabla.sum(axis=1), axis=0)  # Normalizamos por fila (clase_accidente)

    tabla_norm.plot(kind='bar', stacked=True, figsize=(10, 6), colormap='viridis')
    plt.title(f'Stacked Bar Plot: clase_accidente vs {var}')
    plt.xlabel('Clase de Accidente')
    plt.ylabel('Proporción')
    plt.legend(title=var, bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    plt.show()

    #ac_filtrado.to_csv(r"C:\AccidentalidadColombiaAnalisis\Data\Processed\dataset_accidentes_limpio.csv", index=False)