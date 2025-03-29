import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
from plotly.subplots import make_subplots
import geopandas as gpd
from shapely.geometry import Point
import joblib
from datetime import time
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin

df_usagers = pd.read_csv(r"C:\Users\macha\Desktop\Data_scientest\Projet_accidents\Streamlit\all_data_usagers_2019_2023.csv")
df_caracteristiques = pd.read_csv(r"C:\Users\macha\Desktop\Data_scientest\Projet_accidents\Streamlit\all_data_carac_2019_2023.csv")
df_lieux= pd.read_csv(r"C:\Users\macha\Desktop\Data_scientest\Projet_accidents\Streamlit\all_data_lieux_2019_2023.csv")
df_vehicules=pd.read_csv(r"C:\Users\macha\Desktop\Data_scientest\Projet_accidents\Streamlit\all_data_vehicules_2019_2023.csv")
df_total_final = pd.read_csv(r"C:\Users\macha\Desktop\Data_scientest\Projet_accidents\Streamlit\df_total_final.csv")
df_machine_learning =pd.read_csv(r"C:\Users\macha\Desktop\Data_scientest\Projet_accidents\Streamlit\df_machine_learning.csv")
df_usagfinal = pd.read_csv(r"C:\Users\macha\Desktop\Data_scientest\Projet_accidents\Streamlit\df_usagfinal.csv")
df_total_final = pd.read_csv(r"C:\Users\macha\Desktop\Data_scientest\Projet_accidents\Streamlit\df_total_final.csv")
df_lieux_final = pd.read_csv(r"C:\Users\macha\Desktop\Data_scientest\Projet_accidents\Streamlit\df_lieux_final.csv")
df_carac_lieux = pd.read_csv(r"C:\Users\macha\Desktop\Data_scientest\Projet_accidents\Streamlit\df_carac_lieux.csv")
df_merged_usag_veh_final = pd.read_csv(r"C:\Users\macha\Desktop\Data_scientest\Projet_accidents\Streamlit\df_merged_usag_veh_final.csv")
def afficher_infos_dataframe(df):
    st.write(f"**Nombre de lignes :** {df.shape[0]}")
    st.write(f"**Nombre de colonnes :** {df.shape[1]}")


    # Afficher le nombre de valeurs manquantes
    st.write("Valeurs manquantes")
    valeurs_manquantes = df.isnull().sum()
    valeurs_manquantes = valeurs_manquantes[valeurs_manquantes > 0]  # Garde uniquement les colonnes avec valeurs manquantes


    if not valeurs_manquantes.empty:
        st.dataframe(valeurs_manquantes)  # Affiche uniquement si des valeurs manquantes existent
    else:
        st.write("✅ Aucune valeur manquante dans ce dataset")  # Affichage textuel sinon
    
class CyclicalFeatures(BaseEstimator, TransformerMixin):
        def __init__(self, period=24):
                    self.period = period

        def fit(self, X, y=None):
                    return self

        def transform(self, X):
                    X = X.astype(float)  # Assurer que les valeurs sont numériques
                    X_sin = np.sin(2 * np.pi * X / self.period)
                    X_cos = np.cos(2 * np.pi * X / self.period)
                    return np.c_[X_sin, X_cos]  # Retourne un array 2D avec sin et cos


st.sidebar.title("Projet accidents de la route en France")
pages=["Présentation du projet", "Exploration des données", "DataVizualization", "Modélisation","Conclusion"]

if "page" not in st.session_state:
    st.session_state.page = "Présentation du projet"
if "show_exploration" not in st.session_state:
    st.session_state.show_exploration = False
if "show_visualisation" not in st.session_state:
    st.session_state.show_visualisation = False
if "show_modelisation" not in st.session_state:
    st.session_state.show_modelisation = False
if "show_conclusion" not in st.session_state:
    st.session_state.show_conclusion = False

# Fonction pour changer de page
def switch_page(new_page):
    st.session_state.page = new_page

# Fonction pour basculer l'affichage des sections
def toggle_section(section):
    # Réinitialiser toutes les sections sauf celle cliquée
    if section == "exploration":
        st.session_state.show_exploration = not st.session_state.show_exploration
        st.session_state.show_visualisation = False
        st.session_state.show_modelisation = False
        st.session_state.show_conclusion = False
    elif section == "visualisation":
        st.session_state.show_visualisation = not st.session_state.show_visualisation
        st.session_state.show_exploration = False
        st.session_state.show_modelisation = False
        st.session_state.show_conclusion = False
    elif section == "modelisation":
        st.session_state.show_modelisation = not st.session_state.show_modelisation
        st.session_state.show_exploration = False
        st.session_state.show_visualisation = False
        st.session_state.show_conclusion = False
    elif section == "conclusion":
        st.session_state.show_conclusion = not st.session_state.show_conclusion
        st.session_state.show_exploration = False
        st.session_state.show_visualisation = False
        st.session_state.show_modelisation = False

# Gestion de l'état de la page avec session_state
if "page" not in st.session_state:
    st.session_state.page = "Présentation du projet"
if "show_exploration" not in st.session_state:
    st.session_state.show_exploration = False
if "show_visualisation" not in st.session_state:
    st.session_state.show_visualisation = False
if "show_modelisation" not in st.session_state:
    st.session_state.show_modelisation = False
if "show_conclusion" not in st.session_state:
    st.session_state.show_conclusion = False

############################################################ SIDEBAR ##########################################################################################################################################################

st.sidebar.button("Présentation du projet", on_click=switch_page, args=("Présentation du projet",))


######## Exploration des données
# Exploration des données avec effet toggle
if st.sidebar.button("Exploration des données", on_click=toggle_section, args=("exploration",)):
    switch_page("Exploration des données")

# Affichage des sous-sections sous "Exploration des données" avec checkboxes
if st.session_state.show_exploration:
    presentation_donnees_checkbox = st.sidebar.checkbox("Préparation et nettoyage des données")
    elaboration_dataset_checkbox = st.sidebar.checkbox("Dataset final")
  
else:
    presentation_donnees_checkbox = None
    elaboration_dataset_checkbox = None
   
 

############ Data visualisation
# Data Visualisation avec effet toggle
if st.sidebar.button("Data Visualisation", on_click=toggle_section, args=("visualisation",)):
    switch_page("Data Visualisation")

# Affichage des sous-sections sous "Data Visualisation" avec checkboxes
if st.session_state.show_visualisation:
    distribution_gravité_checkbox = st.sidebar.checkbox("Distribution de la variable gravité")
    conditions_accident_checkbox = st.sidebar.checkbox("Conditions de l'accident")
    usagers_checkbox = st.sidebar.checkbox("Usagers impliqués")
    localisation_checkbox = st.sidebar.checkbox("Lieu de l'accident")
    vehicules_checkbox = st.sidebar.checkbox("Véhicules impliqués")
    matrices_corrélation_checkbox = st.sidebar.checkbox("Matrices de corrélation")
else: 
    distribution_gravité_checkbox = None
    conditions_accident_checkbox = None
    usagers_checkbox = None
    localisation_checkbox = None
    vehicules_checkbox = None
    matrices_corrélation_checkbox = None

############## Modélisation  
# Modélisation
if st.sidebar.button("Modélisation", on_click=toggle_section, args=("modelisation",)):
    switch_page("Modélisation")

# Affichage des sous-sections sous "Modélisation" avec checkboxes
if st.session_state.show_modelisation:
    Méthodologie_et_résultats_checkbox = st.sidebar.checkbox("Méthodologie et résultats")
    Prédictions_checkbox = st.sidebar.checkbox("Prédictions")

############## Conclusion

# Affichage des sous-sections sous "Conclusion" avec checkboxes
if st.sidebar.button("Conclusion", on_click=toggle_section, args=("conclusion",)):
    switch_page("Conclusion")




######################################################################################################################################################################################################################""

############################################################################################################# Page présentation du projet #########################################################################################################
if st.session_state.page == "Présentation du projet":
    
    tab1, tab2 = st.tabs(["Projet", " Equipe"])

# Contenu du premier onglet
    with tab1:
        st.header("Projet")
        st.write("L'objectif de ce projet est de prédire la gravité des accidents routiers en France en se basant sur les données historiques recensées en France entre 2005 et 2023. Après réflexion collective, nous décidons de créer un modèle dont l'objectif serait de permettre à un centre d'appel de secours de déterminer le niveau de gravité d'un accident qui vient de se produire, afin de déclencher le nombre et le type de secours adapté. La variable cible est la gravité.Dans cette optique, nous prenons deux décisions:")
        st.write ("Après réflexion collective, nous décidons de créer un modèle dont l'objectif serait de permettre à un centre d'appel de secours de déterminer le niveau de gravité d'un accident qui vient de se produire, afin de déclencher le nombre et le type de secours adapté. La variable cible est la gravité de l'accident")


# Contenu du deuxième onglet
    with tab2:
        st.header("Equipe")
        st.write(" Appolinaire Allarassem")
        st.write("Juliette Meunier")
        st.write("Macha Lagune")

#################################### Page exploration de données et ses sous catégories ####################################
elif st.session_state.page == "Exploration des données":
    st.header("Exploration des données")
    st.write("L'objectif est de fusionner les 4 datasets proposés, en conservant un maximum d'informations, et de commencer à nettoyer les données : Regroupement de certaines valeurs dans une même catégorie, remplacement des valeurs aberrantes, suppression des variables inutiles pour notre problématique")
    st.write("Nous décidons de supprimer les variables suivantes, que nous jugeons non pertinentes pour notre problématique:'lum', 'com', 'adr', 'voie', 'v1', 'v2', 'circ','vosp', 'prof', 'pr', 'pr1', 'plan', 'lartpc', 'larrout', 'infra', 'situ', 'id_vehicule', 'num_veh', 'senc', 'occutc', 'id_usager', 'trajet', 'locp', 'actp', 'etatp'")
#"Préparation des données"
    if presentation_donnees_checkbox:
        st.subheader('Préparation et nettoyage des données')
        # Définition des options
        rubriques = [
    "Rubrique USAGERS","Rubrique VÉHICULES", "Rubrique LIEUX","Rubrique CARACTÉRISTIQUES"]

# Stocker les rubriques sélectionnées
        rubriques_selectionnees = [rubrique for rubrique in rubriques if st.checkbox(rubrique)]

        if "Rubrique USAGERS" in rubriques_selectionnees:
            st.write("### Usagers")
            st.write("La première étape est le travail sur la table Usagers qui est celle avec le niveau d'agrégation le plus fin (1 ligne par usager, total 619971 lignes)")
            st.dataframe(df_usagers)
        # Checkboxes pour afficher les infos
            if st.checkbox("Afficher infos et valeurs manquantes (USAGERS)"):
                afficher_infos_dataframe(df_usagers)
        
            st.write (" ##### Variables transformées et nettoyées dans ce dataset")
            options = ['']+['age']
            choix_usagers = st.selectbox('sélectionner une variable', options)
        
            if 'age' in choix_usagers:
                st.write("##### Age")
                df_usagers["annee"]=df_usagers["Num_Acc"].apply(lambda x: str(x)[:4])   # je rajoute une colonne "année"
                df_usagers["annee"]=df_usagers["annee"].astype(int)
                df_usagers["age"]=df_usagers['annee']-df_usagers['an_nais']# je rajoute une colonne année de naissance de l'usager
                fig, ax = plt.subplots(figsize=(8, 6))
                sns.boxplot(data=df_usagers, x='age', palette="viridis", ax=ax)
                # Ajout du titre
                plt.title("Distribution de l'âge des usagers accidentés")
                st.pyplot(fig)
                age_median = df_usagers['age'].median()
                st.write("L'age médian est")
                st.write(age_median)
                Q1 = df_usagers['age'].quantile(0.25)  # Premier quartile (25e percentile)
                Q3 = df_usagers['age'].quantile(0.75)  # Troisième quartile (75e percentile)
                IQR = Q3 - Q1                  # IQR (Interquartile Range)
                # Définir les limites pour identifier les outliers
                upper_limit = Q3 + 1.5 * IQR
                st.write ('la upper limite est ')
                st.write(upper_limit)
                mode_value = df_usagers['age'].mode()[0]
                st.write('le mode de la variable age est')
                st.write(mode_value)
                st.write("Nous créons ensuite des tranches d'âge , dont une supérieure à la upper limit de 96 ans, pour pourvoir supprimer cette tranche après encodage sans supprimer les autres informations des accidents concernés. Nous remplacerons les valeurs manquantes par le mode de la variable age ")
                st.write("Les tranches d'âge sont ['0-17'], ['18-35'], ['35-61'],['61-95'], ['>96] ")
       
            st.write("Nous obtenons le  dataframe usager final suivant, qui compte désormais une ligne par véhicule")
            st.dataframe(df_usagfinal)
            if st.checkbox("Afficher infos et valeurs manquantes df_usagfinal)"):
                afficher_infos_dataframe(df_usagfinal)
           
        
        if "Rubrique VÉHICULES" in rubriques_selectionnees:  
            st.write("### Véhicules")
            st.write("La 2ème étape consiste à travailler sur la table véhicules afin de pouvoir la merger à la table USAGERS précédente qui compte 1 ligne par véhicule")
            st.dataframe(df_vehicules)
            if st.checkbox("Afficher infos et valeurs manquantes (VEHICULES)"):
                afficher_infos_dataframe(df_vehicules)
            
            st.write (" ##### Variables transformées et nettoyées dans ce dataset")
            options = [''] + ['catv', 'choc']
            choix_véhicules = st.selectbox ('sélectionner une variable', options)
        
            if 'catv' in choix_véhicules:
                st.write("##### Catégorie de véhicule")
                catv_counts = df_vehicules.catv.value_counts()
                fig, ax = plt.subplots(figsize=(4, 3))

                # Tracer un graphique en barres
                ax.bar(catv_counts.index, catv_counts.values)

                # Ajouter un titre et des labels
                ax.set_title("Distribution des valeurs de la variable 'catv'")
                ax.set_xlabel("catv")
                ax.set_ylabel("Nombre")
                # Afficher la figure dans Streamlit
                st.pyplot(fig)
                st.write("Nous décidons de regrouper cette variables en 5 catégories: ")
                def replace_catv(catv):
                    if catv in [7,10]:
                        return 1
                    if catv in [2,30,31,32,33,34,35,36,41,42,43]:
                        return 2
                    if catv in [13,14,15]:
                        return 3
                    if catv in [37,38]:
                        return 4
                    else:
                        return 5


                df_vehicules['new_catv'] = df_vehicules['catv'].apply(replace_catv)
                fig, ax = plt.subplots(figsize=(8, 6))

                sns.countplot(data=df_vehicules, x='new_catv',ax = ax,palette="viridis")
                plt.xticks(ticks=[0,1, 2, 3, 4], labels=["VL/VU", "2-3roues&quad", "PL", "Bus/car", "Velo/EDP/autre"],rotation=45);
                plt.title("Distribution du nombre d'accident par type de véhicule regroupés en 5 catégories")
                st.pyplot(fig)
            
            if 'choc' in choix_véhicules:
                st.write("##### Point de choc")
                valeurs_uniques = df_vehicules['choc'].unique()
                st.write ("Les valeurs uniques de la variable 'catr' sont")
                st.write(*valeurs_uniques)
                st.write("-1 – Non renseigné", "0 – Aucun","1 – Avant","2 – Avant droit","3 – Avant gauche","4 – Arrière","5 – Arrière droit","6 – Arrière gauche","7 – Côté droit","8 – Côté gauche","9 – Chocs multiples (tonneaux)")
                st.write("Nous les regroupons en 5 catégories: ' aucun choc', 'choc_AV, 'choc_AR', 'choc_tonneaux','choc_coté")
        
            st.write("Nous mergeons ensuite cette table à la table usagers précédente, et on obtient le dataframe df_usag_veh suivant, qui compte désormais une ligne par accident, soit 273226  lignes")
            st.dataframe(df_merged_usag_veh_final)
            if st.checkbox("Afficher infos et valeurs manquantes df_merged_usag_veh_final)"):
                afficher_infos_dataframe(df_merged_usag_veh_final)


        if "Rubrique LIEUX" in rubriques_selectionnees:
            st.write("### Lieux")
            st.write("La 3ème étape est le travail sur la table lieux")
            st.dataframe(df_lieux)
            if st.checkbox("Afficher infos et valeurs manquantes (LIEUX)"):
                afficher_infos_dataframe(df_lieux)
            
            st.write (" ##### Variables transformées et nettoyées dans ce dataset")
            options = [''] + ['nbv','vma','catr', 'circ', 'surf']
            choix_lieux = st.selectbox ('sélectionner une variable', options)
        
            if 'nbv' in choix_lieux:
                st.write("##### Nombre de voies")
                mode_value = df_total_final['nbv'].mode()[0]
                st.write ('Le mode de la variable nbv est')
                st.write(mode_value)
                df_total_final['nbv']= df_total_final['nbv'].replace([-1, 11,12, '-1', ' -1', 0], mode_value)
                df_total_final['nbv'] = df_total_final['nbv'].replace(['#VALEURMULTI', '-1', '0', '#ERREUR', '11', '12'], mode_value)
                df_total_final['nbv'] = df_total_final['nbv'].replace({'2': 2, '8': 8, '6': 6, '4': 4, '5': 5, '7': 7, '3':3, '1':1, '10':10, '9':9})

                fig, ax = plt.subplots(figsize=(4, 3))  # Taille de la figure

                # Utiliser sns.countplot pour afficher les données sur ax
                sns.countplot(data=df_total_final, x='nbv', ax=ax)

                # Ajouter un titre au graphique
                ax.set_title("Distribution des Nombre de Voies")

                # Afficher la figure sur Streamlit
                st.pyplot(fig)  # Afficher la figure
            
            if 'vma' in choix_lieux:
                st.write("##### Vitesse maximale autorisée")
                valeurs_uniques = df_lieux['vma'].unique()
                st.write ("Les valeurs uniques de la variable 'vma' sont")
                st.write(*valeurs_uniques)
                st.write("On constate qu'il y a énormémement de valeurs aberrantes, que l'on décide de remplacer par le mode de la variable vma")
                st.write('Le mode de la variable vma est')
                mode_vma = df_lieux['vma'].mode()[0]
                st.write(mode_vma)
                df_lieux['vma']=df_lieux['vma'].replace([-1,1,2,31,45,5,15,25,10,40,500,6,35,3,300,900,4,65,700,75,7,12,55,8,0,180,140,770,502,501,9,901,520,600,42,800,560,120,23], df_lieux['vma'].mode()[0])
                st.write('Après transformation, les nouvelles valeurs uniques sont')
                new_mode_vma = df_lieux['vma'].unique()
                st.write(*new_mode_vma)

            if 'catr' in choix_lieux:
                st.write("##### Catégorie de route")
                valeurs_uniques = df_lieux['catr'].unique()
                st.write ("Les valeurs uniques de la variable 'catr' sont")
                st.write(*valeurs_uniques)
                st.write ("  1 – Autoroute","  2 – Route nationale", "  3 – Route Départementale", "  4 – Voie Communales","  5 – Hors réseau public","  6 – Parc de stationnement ouvert à la circulation publique","  7 – Routes de métropole urbaine","  9 – autre")
                st.write ('Nous les regroupons en trois catégories: autoroute, nationale_départementale_communale et autre')
                
            if 'circ' in choix_lieux:
                st.write("##### Sens de circulation")
                valeurs_uniques = df_lieux['circ'].unique()
                st.write ("Les valeurs uniques de la variable 'circ' sont")
                st.write( "-1 – Non renseigné", "1 – A sens unique","2 – Bidirectionnelle","3 – A chaussées séparées", "4 – Avec voies d’affectation variable")
                st.write(*valeurs_uniques)
                st.write ("Nous les regroupons en deux catégories: ' Sens unique', et 'bidirectionnel'")

            if 'surf' in choix_lieux:
                st.write("##### Type de surface")
                data = data = df_lieux["surf"].value_counts().reset_index()
                st.write ("La distribution de la variable 'surf' est")
                st.dataframe(data)
                st.write ("-1 – Non renseigné", '1 – Normale', "2 – Mouillée","3 – Flaques", "4 – Inondée","5 – Enneigée", "6 – Boue"," 7 – Verglacée","8 – Corps gras-huile", "9 – Autre")
                st.write ('Nous les regroupons en trois catégories: #1 normale, #2 mouillée/enneigee , #3 autre')
                
            st.write ("On obtient le dataframe lieux final suivant")
            st.dataframe(df_lieux_final)
            if st.checkbox("Afficher infos et valeurs manquantes df_lieux_final)"):
                afficher_infos_dataframe(df_lieux_final)
        
        if "Rubrique CARACTÉRISTIQUES" in rubriques_selectionnees:
            st.write("### Caractéristiques")
            st.write("Nous travaillons enfin sur la table caractéristiques qui contient une ligne par accident, soit 273226 lignes")
            st.dataframe(df_caracteristiques)
            if st.checkbox("Afficher infos et valeurs manquantes (CARACTÉRISTIQUES)"):
                    afficher_infos_dataframe(df_caracteristiques)

            st.write (" ##### Variables transformées et nettoyées dans ce dataset")
            options = [''] + ['atmosphère']
            choix = st.selectbox ('sélectionner une variable', options)
        
            if 'atmosphère' in choix:
                st.write("##### Atmosphère")
                atm_counts = df_total_final.atm.value_counts()
                fig, ax = plt.subplots(figsize=(10, 6))

                # Tracer un graphique en barres
                ax.bar(atm_counts.index, atm_counts.values)

                # Ajouter un titre et des labels
                ax.set_title("Distribution des valeurs de la variable atmosphère")
                ax.set_xlabel("Atmosphère")
                ax.set_ylabel("Nombre")

                # Afficher la figure dans Streamlit
                st.pyplot(fig)

                def regrouper(val):
                    if val in [1]:
                        return 'temps_normal'
                    elif val in [2, 3]:
                        return 'Temps_pluvieux'
                    elif val in [8]:
                        return 'Temps_couvert'
                    else:
                        return 'Autre'

                df_total_final['atm'] = df_total_final['atm'].apply(regrouper)
                valeurs_uniques = df_total_final.atm.unique()
                st.write('Après regroupement, les valeurs uniques de la variable atmosphère sont les suivantes' )
                st.write(*valeurs_uniques)
            
            st.write("Nous mergeons ensuite les dataframe lieux et caractéristiques, et obtenons le dataframe df_lieux_carac suivant")
            st.dataframe(df_carac_lieux)
            if st.checkbox("Afficher infos et valeurs manquantes df_carac_lieux)"):
                afficher_infos_dataframe(df_carac_lieux)
        
        
# Elaboration dataset      

    if elaboration_dataset_checkbox:
        st.header("Elaboration du dataset final")
        st.write("Nous pouvons enfin merger les dataframes df_usag_veh et df_carac _lieux pour obtenir le dataframe final")
        # Afficher le dataframe des 20 premières lignes
        st.write("### Aperçu du Dataset Final")
        st.dataframe(df_total_final.head(20))
        st.write("Beaucoup de variables ont été encodées lors de la fusion des 4 dataframes, nous pourrons donc supprimer les colonnes 'variable_non_déterminé' pour les exclure sans perdre les autres inforamtions des accidents, usagers, véhicules")


        

        

######################################### Page Datavisualisation et ses sous rubriques ################################
elif st.session_state.page == "Data Visualisation":
    st.header('Data Visualisation')
    st.write("L'objectif de cette partie est de s'approprier les données et de déterminer l'importance de chaque variable par rapport à la gravité de l'accident.")

# Analyse de la variable gravité
    if distribution_gravité_checkbox:
        st.subheader('Gravité des accidents')
        df_total_final['date']= pd.to_datetime(df_total_final['jour'].astype('str')+'/'+df_total_final['mois'].astype('str')+'/'+df_total_final['an'].astype('str')+ '/'+df_total_final['hrmn'].astype('str'), dayfirst = True)
        gravité = df_total_final.groupby('gravité_accident').size().reset_index(name='nombre_accidents')
        fig = px.bar(gravité, x='gravité_accident', y = 'nombre_accidents',title="Répartition de la gravité", color = 'gravité_accident')
        fig.update_xaxes(tickmode='array', tickvals=[2,3,4],ticktext=['blessé_léger', 'blessé_grave', 'tué'])
        st.plotly_chart(fig)

        st.write(" Nous observons un déséquilibre de la répartition des valeurs de la variable cible (gravité de l'accident), qu'il faudra garder en tête pour la phase de modélisation")
        st.write("Nous constatons également qu’il n’y a aucun accident présentant la gravité 'indemne', ce qui signifie que pour chaque accident répertorié, il y a au moins un blessé.")


# Conditions accident
    if conditions_accident_checkbox:
        st.subheader('Conditions des accidents')
        variables_conditions_accident = st.radio(
        "Sélectionnez une variable :",
        ["Date de l'accident", "Vitesse maximale autorisée"])
       
        if "Date de l'accident" in variables_conditions_accident:
            st.write("### Date de l'accident")
            df_total_final['date']= pd.to_datetime(df_total_final['jour'].astype('str')+'/'+df_total_final['mois'].astype('str')+'/'+df_total_final['an'].astype('str')+ '/'+df_total_final['hrmn'].astype('str'), dayfirst = True)
            fig = make_subplots(rows=2, cols=2, subplot_titles=("Nombre d'accidents par année",  "Nombre d'accidents par jour de la semaine", "Nombre d'accidents par mois","Nombre d'accidents par heure"))
            accidents_par_an = df_total_final.groupby([df_total_final['date'].dt.year,'gravité_accident']).size().reset_index(name='nombre_accidents_an')
            accidents_par_an['date'] = accidents_par_an['date'].astype(str)
            bar1 = px.bar(accidents_par_an, y='nombre_accidents_an', x='date', color = 'gravité_accident', barmode = 'group',labels={"gravité_accident": "Gravité de l'accident", "nombre_accidents_an": "Nombre d'accidents"},  # Labels personnalisés
            title="Nombre d'accidents par gravité et par année",)
            for trace in bar1.data:
                fig.add_trace(trace, row=1, col=1)
                fig.update_xaxes(tickmode='array', tickvals=[2019, 2020, 2021, 2022,2023],ticktext=['2019', '2020', '2021', '2022','2023'], row=1, col=1)


                accidents_par_mois = df_total_final.groupby([df_total_final['date'].dt.month, 'gravité_accident']).size().reset_index(name='nombre_accidents_mois')
                bar2 = px.bar(accidents_par_mois, x='date', y='nombre_accidents_mois',color = 'gravité_accident')
                for trace in bar2.data:
                    fig.add_trace(trace, row=2, col=1)
                    fig.update_xaxes(tickmode='array', tickvals=[1,2,3,4,5,6,7,8,9,10,11,12],  # Exemple: vos années
                    ticktext=['janvier','fevrier', 'mars', 'avril', 'mai', 'juin', 'juillet', 'aout', 'septembre', 'octobre', 'novembre', 'décembre'], row=2, col=1)



                accidents_par_jour = df_total_final.groupby([df_total_final['date'].dt.weekday, 'gravité_accident']).size().reset_index(name='nombre_accidents_jour')
                bar3 = px.bar(accidents_par_jour, x='date', y='nombre_accidents_jour',color = 'gravité_accident')
                for trace in bar3.data:
                    fig.add_trace(trace, row=1, col=2)
                    fig.update_xaxes(tickmode='array', tickvals=[0,1,2,3,4,5,6],ticktext = ['Lundi', 'Mardi', 'Mercredi', 'Jeudi', 'Vendredi', 'Samedi', 'Dimanche'], row=1, col=2)

                accidents_par_heure = df_total_final.groupby([df_total_final['date'].dt.hour, 'gravité_accident']).size().reset_index(name='nombre_accidents_heure')
                bar4 = px.bar(accidents_par_heure, x='date', y='nombre_accidents_heure',color = 'gravité_accident')
                for trace in bar4.data:
                    fig.add_trace(trace, row=2, col=2)
                    fig.update_xaxes(tickmode='array', tickvals=[0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23],ticktext =  ['0', '1','2','3','4','5','6','7','8', '9', '10','11','12','13','14','15','16','17','18', '19', '20', '21', '22','23'], row=2, col=2)

                fig.update_layout(title="Répartition des accidents par année, mois et jour de la semaine", showlegend=False,barmode = 'group', hovermode='y')
                st.plotly_chart(fig)  

                # Camembert par jour de semaine
                df_total_final["jour_semaine"]=df_total_final["date"].dt.weekday
                fig2, ax = plt.subplots()
                ax.pie(df_total_final["jour_semaine"].value_counts(), labels=["Lundi","Mardi","Mercredi","Jeudi","Vendredi","Samedi","Dimanche"], autopct="%1.1f%%")
                # Ajout du titre
                plt.title("Répartition des accidents par jour de la semaine")

                st.write("Le nombre d'accidents par an est à peu près stable, sauf l'année 2020 qui est l'année du covid. Il y a plus d'accidents en juin, juillet, septembre, octobre, et le vendredi, et entre 17 et 18h.")
            

          

        if "Vitesse maximale autorisée" in variables_conditions_accident:
            st.write("### Vitesse maximale autorisée")

            # Filtrage des données (assurez-vous que df_total_final est bien défini)
            df_filtered = df_total_final[df_total_final['vma'].isin([30, 50, 70, 90, 110, 130])]

            # Création de la figure Matplotlib
            fig, ax = plt.subplots()
            sns.countplot(x='vma', hue='gravité_accident', data=df_filtered, ax=ax)
            ax.set_title("Répartition du nombre d'accidents selon la vitesse maximale autorisée et la gravité")
            ax.set_xticklabels(ax.get_xticklabels(), rotation=90)
            ax.set_xlabel("Vitesse maximale autorisée")

            # Affichage du graphique dans Streamlit
            st.pyplot(fig)


            # Regrouper les données par gravité pour chaque vitesse
            vitesses = [30, 50, 70, 90, 110, 130]
            gravite_labels = ['Blessés légers', 'Blessés hospitalisés', 'Tués']
            
            # Création d'une figure avec 6 sous-graphiques
            fig, axs = plt.subplots(1, 6, figsize=(18, 6))

            colors = ["green", "orange", "red"]

            for i, vitesse in enumerate(vitesses):
                data = df_filtered[df_filtered['vma'] == vitesse].groupby('gravité_accident').size()

                if not data.empty:  # Vérifier qu'il y a bien des données avant d'afficher
                    axs[i].pie(data, colors=colors, autopct="%1.1f%%")
                    axs[i].set_title(f"Vitesse {vitesse} km/h")
                else:
                    axs[i].set_visible(False)  # Masquer l'axe s'il n'y a pas de données

            fig.suptitle("Répartition des accidents par gravité pour différentes vitesses")

            # Affichage du graphique dans Streamlit
            st.pyplot(fig)

            

    ## Usagers
    if usagers_checkbox:
        st.subheader("Usagers impliqués")
        variables_usagers = st.radio(
        "Sélectionnez une variable :",
        ["Sexe", "Age", "Système de sécurité"])
       
        if "Sexe" in variables_usagers:
            # Fonction pour agréger les données par sexe et gravité des accidents avec cache
            @st.cache_data
            def aggregate_sex_data(df):
                return df.groupby('gravité_accident').agg({'homme':'sum', 'femme': 'sum'}).reset_index()

            # Fonction pour agrégée les données par gravité pour les hommes et les femmes avec cache
            @st.cache_data
            def aggregate_data(sex):
                grav_homme = sex[['gravité_accident', 'homme']].groupby('gravité_accident')['homme'].sum().reset_index()
                grav_femme = sex[['gravité_accident', 'femme']].groupby('gravité_accident')['femme'].sum().reset_index()
                return grav_homme, grav_femme

            # Fonction pour créer le graphique à barres avec cache
            @st.cache_data
            def create_bar_chart(sex):
                fig = px.bar(sex, x='gravité_accident', y=['homme', 'femme'], title="Gravité selon le sexe", barmode='group')
                fig.update_xaxes(tickmode='array', tickvals=[2, 3, 4], ticktext=['blessé_léger', 'blessé_grave', 'tué'])
                return fig

            # Fonction pour créer les graphiques en camembert avec cache
            @st.cache_data
            def create_pie_charts(grav_homme, grav_femme):
                # Labels des secteurs
                labels = ['Blessés légers', 'Blessés hospitalisés', 'Tués']
                
                # Création des graphiques pie charts
                fig, axs = plt.subplots(1, 2, figsize=(20, 8))

                # Graphique pie chart pour les hommes
                axs[0].pie(grav_homme['homme'], labels=grav_homme['gravité_accident'], colors=["green", "orange", "red"], autopct="%1.1f%%")
                axs[0].set_title("Hommes")

                # Graphique pie chart pour les femmes
                axs[1].pie(grav_femme['femme'], labels=grav_femme['gravité_accident'], colors=["green", "orange", "red"], autopct="%1.1f%%")
                axs[1].set_title("Femmes")

                # Titre global
                fig.suptitle("Répartition des accidents par gravité chez les hommes et chez les femmes")

                # Légende
                fig.legend(labels, loc='upper left')

                return fig

            # Agrégation des données avec cache
            sex = aggregate_sex_data(df_total_final)

            # Affichage du titre
            st.write("### Sexe")

            # Création et affichage du graphique à barres avec cache
            fig_bar = create_bar_chart(sex)
            st.plotly_chart(fig_bar)

            # Agrégations pour les hommes et les femmes
            grav_homme, grav_femme = aggregate_data(sex)

            # Création et affichage des graphiques en camembert avec cache
            fig_pie = create_pie_charts(grav_homme, grav_femme)
            st.pyplot(fig_pie)

            # Conclusion
            st.write("Les hommes ont plus d'accidents que les femmes, et la proportion d'accidents mortels ou graves chez les hommes est aussi plus élevée que chez les femmes. La variable sexe semble influencer la gravité de l'accident.")

          

        if "Age" in variables_usagers:
            @st.cache_data
            def aggregate_age_data(df):
                return df.groupby('gravité_accident').agg({'0-17': 'sum', '18-60': 'sum', '61-95': 'sum'}).reset_index()

            # Fonction pour créer le graphique bar et le mettre en cache
            @st.cache_data
            def create_age_bar_chart(age):
                fig = px.bar(age, x='gravité_accident', y=['0-17', '18-60', '61-95'], 
                            title="Répartition du nombre d'usagers par classe d'âge et par gravité", barmode='group')
                fig.update_xaxes(tickmode='array', tickvals=[2, 3, 4], ticktext=['blessé_léger', 'blessé_grave', 'tué'])
                return fig

    # Fonction pour créer les graphiques pie charts et les mettre en cache
            @st.cache_data
            def create_pie_charts(age):
                grav_0_17 = age[['gravité_accident', '0-17']]
                grav_18_60 = age[['gravité_accident', '18-60']]
                grav_61_95 = age[['gravité_accident', '61-95']]
                
                labels = ['Blessés légers', 'Blessés hospitalisés', 'Tués']
                fig, axs = plt.subplots(1, 3, figsize=(20, 8))

                # Graphique pie chart pour les 0-17 ans
                axs[0].pie(grav_0_17['0-17'], labels=grav_0_17['gravité_accident'], colors=["green", "orange", "red"], autopct="%1.1f%%")
                axs[0].set_title("0-17 ans")

                # Graphique pie chart pour les 18-60 ans
                axs[1].pie(grav_18_60['18-60'], labels=grav_18_60['gravité_accident'], colors=["green", "orange", "red"], autopct="%1.1f%%")
                axs[1].set_title("18-60 ans")

                # Graphique pie chart pour les 61-95 ans
                axs[2].pie(grav_61_95['61-95'], labels=grav_61_95['gravité_accident'], colors=["green", "orange", "red"], autopct="%1.1f%%")
                axs[2].set_title("61-95 ans")

                fig.suptitle("Répartition des accidents par gravité pour les différentes tranches d'âge")
                fig.legend(labels, loc='upper left')
                
                return fig


            # Agrégation des données avec cache
            age = aggregate_age_data(df_total_final)

            # Affichage du titre
            st.write("### Age")

            # Création et affichage du graphique bar avec cache
            fig_bar = create_age_bar_chart(age)
            st.plotly_chart(fig_bar)

            # Création et affichage des graphiques pie avec cache
            fig_pie = create_pie_charts(age)
            st.pyplot(fig_pie)

            st.write("IL y a évidemment beaucoup plus de 18-60 ans impliqués dans les accidents puisque la tranche d'âge est plus large. On constate que la proportion d'accidents graves (tués et blessés hospitalisés) est plus élevée chez les 61-96 ans. L'age semble donc impacter la gravité de l'accident.")
        
        if "Système de sécurité" in variables_usagers:
            
            # Fonction pour agréger les données avec cache
            @st.cache_data
            def aggregate_gravity_data(df):
                return df.groupby('gravité_accident').agg({
                    'total_sans_secu': 'sum', 
                    'total_ceinture': 'sum', 
                    'total_casque': 'sum', 
                    'total_secu_enfant': 'sum', 
                    'total_gilet': 'sum', 
                    'total_airbag': 'sum', 
                    'total_gants': 'sum', 
                    'total_gants_airbag': 'sum', 
                    'total_autre': 'sum'
                }).reset_index()

            # Fonction pour créer le graphique avec cache
            @st.cache_data
            def create_bar_chart(grav):
                fig = px.bar(
                    grav, 
                    x='gravité_accident', 
                    y=['total_sans_secu', 'total_ceinture', 'total_casque', 'total_secu_enfant', 'total_gilet', 'total_airbag', 'total_gants', 'total_gants_airbag', 'total_autre'], 
                    title="Répartition du nombre d'usagers par catégorie de système de sécurité selon la gravité", 
                    barmode='group'
                )
                fig.update_xaxes(tickmode='array', tickvals=[2, 3, 4], ticktext=['blessé_léger', 'blessé_grave', 'tué'])
                return fig

            # Agrégation des données avec cache
            grav = aggregate_gravity_data(df_total_final)

            # Affichage du titre
            st.write("### Système de sécurité")

            # Création du graphique avec cache
            fig_bar = create_bar_chart(grav)

            # Affichage du graphique
            st.plotly_chart(fig_bar)

            st.write("On constate que l'équipement de sécurité majoritaire est le port de ceinture, et que l'absence de ceinture arrive en seconde position, quelle que soit la gravité de l'accident.")

# Lieu 
    if localisation_checkbox:
        st.subheader('Lieux des accidents')
        variables_lieux = st.radio(
        "Sélectionnez une variable :",
        ["Carte géographique de la répartition des accidents", "Agglomération/hors_agglomération"])
       
        if "Carte géographique de la répartition des accidents" in variables_lieux:
            @st.cache_data
            def process_accident_data(df):
                # Transformation des coordonnées
                df['lat'] = df['lat'].astype('str').apply(lambda x: x.replace('\t', '.').replace(',', '.'))
                df['lat'] = df['lat'].astype('float').round(5)
                df['long'] = df['long'].astype('str').apply(lambda x: x.replace('\t', '.').replace(',', '.'))
                df['long'] = df['long'].astype('float').round(5)

                # Sélection des lignes avec des départements numériques
                df_dep_num = df.loc[df['dep'].astype('str').apply(lambda x: x.isnumeric())].astype('str')

                # Sélection des lignes avec des départements non numériques (Corse)
                df_dep_corse = df.loc[df['dep'].astype('str').apply(lambda x: x.isnumeric() == False)].astype('str')

                # Sélectionner les accidents en métropole
                df_m = df[(df['dep'].isin(df_dep_num.dep.unique())) | (df['dep'].isin(df_dep_corse.dep.unique()))]

                # Supprimer les accidents avec des coordonnées nulles ou manquantes
                df_m = df_m[(df_m['lat'] != 0.0) & (df_m['long'] != 0.0)].dropna(subset=['lat', 'long'], axis=0)

                # Créer une géométrie Point pour chaque paire de coordonnées (longitude, latitude)
                geometry = [Point(xy) for xy in zip(df_m['long'], df_m['lat'])]

                # Créer un GeoDataFrame à partir des données et de la géométrie
                geo_df = gpd.GeoDataFrame(df_m, geometry=geometry)

                return geo_df

            # Fonction pour afficher la carte avec cache
            @st.cache_data
            def plot_accident_map(_geo_df):
                # Créer une figure et des axes pour la carte
                figure, ax = plt.subplots(figsize=(13, 12))

                # Masquer les axes pour une apparence plus propre
                plt.axis('off')

                # Définir les limites des axes pour inclure toute la France
                ax.set_xlim([-5.5, 9.5])  # Longitude (Ouest-Est)
                ax.set_ylim([41.0, 51.5])  # Latitude (Sud-Nord)

                # Tracer les accidents sur la carte en utilisant différentes couleurs pour chaque gravité
                geo_df[geo_df['gravité_accident'] == 2].plot(ax=ax, markersize=5, color='green', label='Blessés légers')
                geo_df[geo_df['gravité_accident'] == 3].plot(ax=ax, markersize=5, color='orange', label='Blessés graves')
                geo_df[geo_df['gravité_accident'] == 4].plot(ax=ax, markersize=5, color='red', label='Tués')

                # Titre et légende
                plt.title('Représentation de la gravité des accidents en France entre 2019 et 2023')
                plt.legend()

                # Retourner la figure pour affichage
                return figure

            # Traitement des données avec cache
            geo_df = process_accident_data(df_total_final)

            # Affichage dans Streamlit
            st.write("### Carte de la répartition des accidents en France par gravité")
            fig = plot_accident_map(geo_df)

            # Affichage de la carte avec Streamlit
            st.pyplot(fig)

            st.write("'Ile de France regroupe le plus d'accidents mortels. La région PACA et Lyon ont également beaucoup de blessés graves et du tués. On constate une plus forte concentration d'accidents dans les villes")
          

        if "Agglomération/hors_agglomération" in variables_lieux:
            st.write("Agglomération/hors_agglomération")
            
    
            def plot_accident_distribution(df):
                # Créer le graphique avec seaborn
                fig, ax = plt.subplots()
                sns.countplot(x='agg', hue='gravité_accident', data=df, ax=ax)
                ax.set_title("Répartition du nombre d'accidents selon la localisation agglo/hors agglo")
                ax.set_xticklabels(['hors agglomaration', 'en agglomération'])
                ax.set_xlabel('Localisation')
                st.pyplot(fig)  # Affiche le graphique dans Streamlit

            

            plot_accident_distribution(df_total_final)

            st.write("Il y a beaucoup plus d'accidents en agglomération qu'hors agglomération, ce que l'on avait déjà noté sur la carte géographique. Les accidents sont plus graves en dehors des agglomérations. On peut faire l'hypothèse que la localisation(agglo/hors agglo) influence la gravité de l'accident. Vérifions par un test statistique")

########## Véhicules
    if vehicules_checkbox:
        st.subheader("Véhicules impliqués")
        variables_vehicule = st.radio(
        "Sélectionnez une variable :",
        ["Type de véhicule", "Point de choc"])
       
        if "Type de véhicule" in variables_vehicule:
            st.write("Type de véhicule")

            # Fonction de mise en cache pour les données filtrées et les calculs de proportion
            @st.cache_data
            def get_accidents_with_vehicle(df, column_name):
                # Filtrer les accidents où au moins 1 véhicule de type spécifié est impliqué
                accidents = df.loc[df[column_name] >= 1]
                # Calculer la proportion des gravités d'accidents
                accident_by_severity = accidents.groupby("gravité_accident").size()
                return accident_by_severity.div(accident_by_severity.sum())

            # Affichage des résultats et graphiques
            def plot_accident_distribution(acc_avec_pl, acc_avec_2_3roues, acc_avec_bus_car, 
                                            acc_av_vl_vu, acc_av_velo_trott_edp, acc_av_pietons):
                labels = ['Blessés légers', 'Blessés hospitalisés', 'Tués']
                
                fig, axs = plt.subplots(2, 3, figsize=(15, 10))  # 2 lignes et 3 colonnes

                # Graphiques pour la première ligne
                axs[0, 0].pie(acc_avec_pl, colors=["green", "orange", "red"], autopct="%1.1f%%")
                axs[0, 0].set_title("Avec au moins PL")
                axs[0, 1].pie(acc_avec_2_3roues, colors=["green", "orange", "red"], autopct="%1.1f%%")
                axs[0, 1].set_title("Avec au moins un 2 ou 3 roues ou quad")
                axs[0, 2].pie(acc_avec_bus_car, colors=["green", "orange", "red"], autopct="%1.1f%%")
                axs[0, 2].set_title("Avec au moins bus/car")

                # Graphiques pour la deuxième ligne
                axs[1, 0].pie(acc_av_vl_vu, colors=["green", "orange", "red"], autopct="%1.1f%%")
                axs[1, 0].set_title("Avec au moins un VL VU")
                axs[1, 1].pie(acc_av_velo_trott_edp, colors=["green", "orange", "red"], autopct="%1.1f%%")
                axs[1, 1].set_title("Avec au moins un vélo, trott ou edp")
                axs[1, 2].pie(acc_av_pietons, colors=["green", "orange", "red"], autopct="%1.1f%%")
                axs[1, 2].set_title("Avec au moins un piéton")

                fig.suptitle("Répartition des accidents par gravité pour différents types de véhicules")
                fig.legend(labels, loc='center')
                
                # Affichage dans Streamlit
                st.pyplot(fig)

            # Calculer les proportions pour différents types de véhicules
            acc_avec_pl = get_accidents_with_vehicle(df_total_final, "PL")
            acc_avec_2_3roues = get_accidents_with_vehicle(df_total_final, "2roues_3roues_quad")
            acc_avec_bus_car = get_accidents_with_vehicle(df_total_final, "bus_car")
            acc_av_vl_vu = get_accidents_with_vehicle(df_total_final, "VL_VU")
            acc_av_velo_trott_edp = get_accidents_with_vehicle(df_total_final, "velo_trott_edp")
            acc_av_pietons = get_accidents_with_vehicle(df_total_final, "place_pieton")


            # Afficher les graphiques
            plot_accident_distribution(acc_avec_pl, acc_avec_2_3roues, acc_avec_bus_car, 
                                        acc_av_vl_vu, acc_av_velo_trott_edp, acc_av_pietons)

            st.write("Les accidents les plus graves sont ceux impliquant des tonneaux et des 2/3 roues. Les accidents les plus mortels sont ceux impliquant au moins un poids lourd.")

        if "Point de choc" in variables_vehicule:
            st.write("Point de choc")

            # Cache des données traitées
            @st.cache_data
            def get_gravité_accidents(df):
                grav = df.groupby('gravité_accident').agg({
                    'choc_AV':'sum', 
                    'choc_AR': 'sum', 
                    'choc_cote': 'sum', 
                    'choc_tonneaux':'sum', 
                    'aucun_choc':'sum'
                }).reset_index()
                return grav

            @st.cache_data
            def get_accidents_with_choc_type(df, choc_column, condition):
                accidents = df.loc[condition]
                return accidents.groupby("gravité_accident").size()

            # Calculer les agrégations nécessaires pour les graphiques
            grav = get_gravité_accidents(df_total_final)

            # Affichage du graphique avec Plotly
            fig = px.bar(grav, x='gravité_accident', y=['choc_AV', 'choc_AR', 'choc_cote', 'choc_tonneaux', 'aucun_choc'], 
                        title="Répartition du nombre d'accidents selon la gravité et le point de choc", 
                        barmode='group')
            fig.update_xaxes(tickmode='array', tickvals=[2, 3, 4], ticktext=['blessé_léger', 'blessé_grave', 'tué'])

            # Affichage du graphique dans Streamlit
            st.plotly_chart(fig)

            # Extraire les accidents pour chaque type de choc
            acc_sans_choc = get_accidents_with_choc_type(df_total_final, 'aucun_choc', (df_total_final["aucun_choc"] >= 1) & (df_total_final["choc_AV"] == 0) & (df_total_final["choc_AR"] == 0) & (df_total_final["choc_tonneaux"] == 0) & (df_total_final["choc_cote"] == 0))
            acc_av_choc_av = get_accidents_with_choc_type(df_total_final, 'choc_AV', (df_total_final["aucun_choc"] == 0) & (df_total_final["choc_AV"] >= 1) & (df_total_final["choc_AR"] == 0) & (df_total_final["choc_tonneaux"] == 0) & (df_total_final["choc_cote"] == 0))
            acc_av_tonneaux = get_accidents_with_choc_type(df_total_final, 'choc_tonneaux', (df_total_final["aucun_choc"] == 0) & (df_total_final["choc_AV"] == 0) & (df_total_final["choc_AR"] == 0) & (df_total_final["choc_tonneaux"] >= 1) & (df_total_final["choc_cote"] == 0))
            acc_av_choc_ar = get_accidents_with_choc_type(df_total_final, 'choc_AR', (df_total_final["aucun_choc"] == 0) & (df_total_final["choc_AV"] == 0) & (df_total_final["choc_AR"] >= 1) & (df_total_final["choc_tonneaux"] == 0) & (df_total_final["choc_cote"] == 0))
            acc_av_choc_cote = get_accidents_with_choc_type(df_total_final, 'choc_cote', (df_total_final["aucun_choc"] == 0) & (df_total_final["choc_AV"] == 0) & (df_total_final["choc_AR"] == 0) & (df_total_final["choc_tonneaux"] == 0) & (df_total_final["choc_cote"] >= 1))

            # Extraire les accidents avec plusieurs chocs
            acc_chocs_multiples = df_total_final.loc[
                ((df_total_final["choc_AV"] >= 1).astype(int) + (df_total_final["choc_AR"] >= 1).astype(int) + (df_total_final["choc_tonneaux"] >= 1).astype(int) + (df_total_final["choc_cote"] >= 1).astype(int)) >= 2
            ]
            acc_chocs_multiples = acc_chocs_multiples.groupby("gravité_accident").size()

            # Création des graphiques en secteurs
            labels = ['Blessés légers', 'Blessés hospitalisés', 'Tués']
            fig, axs = plt.subplots(2, 3, figsize=(15, 10))

            # Graphiques en secteurs pour chaque type de choc
            axs[0, 0].pie(acc_sans_choc, colors=["green", "orange", "red"], autopct="%1.1f%%")
            axs[0, 0].set_title("Sans choc")
            axs[0, 1].pie(acc_av_choc_av, colors=["green", "orange", "red"], autopct="%1.1f%%")
            axs[0, 1].set_title("Avec choc AV")
            axs[0, 2].pie(acc_av_choc_ar, colors=["green", "orange", "red"], autopct="%1.1f%%")
            axs[0, 2].set_title("Avec choc AR")
            axs[1, 0].pie(acc_av_tonneaux, colors=["green", "orange", "red"], autopct="%1.1f%%")
            axs[1, 0].set_title("Avec tonneaux")
            axs[1, 1].pie(acc_av_choc_cote, colors=["green", "orange", "red"], autopct="%1.1f%%")
            axs[1, 1].set_title("Avec choc coté")
            axs[1, 2].pie(acc_chocs_multiples, colors=["green", "orange", "red"], autopct="%1.1f%%")
            axs[1, 2].set_title("Avec plusieurs chocs")

            # Titre et légende des graphiques
            fig.suptitle("Répartition des accidents par gravité pour différents types de chocs")
            fig.legend(labels, loc='center')

            # Affichage du graphique dans Streamlit
            st.pyplot(fig)


# Matrices de corrélation
    if matrices_corrélation_checkbox:
        st.subheader("Matrices de corrélation")
        variables_matrices_corrélation = st.radio(
        "Sélectionnez une variable :",
        ["Corrélation selon les conditions des usagers", "Corrélation selon les conditions de la route","Corrélation selon les obstacles et types de chocs" ])
       
        if "Corrélation selon les conditions des usagers" in variables_matrices_corrélation:
            @st.cache_data
            def get_filtered_data(df):
                to_drop = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]
                df_filtré = df.drop(df.columns[to_drop], axis=1)
                return df_filtré

            @st.cache_data
            def get_correlation_matrix(df_filtré, selected_columns1):
                corr_matrix_subset1 = df_filtré[selected_columns1].corr()
                return corr_matrix_subset1

            # Sélectionner les colonnes pour la matrice de corrélation
            selected_columns1 = [
                'usager_count', '0-17', '18-60', '61-95', 'gravité_accident', 'homme', 'femme', 
                'place_conducteur', 'pax_AV', 'pax_AR', 'pax_Milieu', 'place_pieton', 
                'blessé_léger', 'blessé_hospitalisé', 'tué'
            ]

            # Appliquer le cache sur les données filtrées et la matrice de corrélation
            df_filtré = get_filtered_data(df_total_final)
            corr_matrix_subset1 = get_correlation_matrix(df_filtré, selected_columns1)

            # Affichage de la matrice de corrélation avec Seaborn
            st.subheader("Matrice de corrélation selon les conditions des usagers")
            # Créer un objet figure explicitement
            fig, ax = plt.subplots(figsize=(12, 8))

            # Tracer la carte de chaleur sur cet axe (ax)
            sns.heatmap(corr_matrix_subset1, annot=True, cmap="coolwarm", linewidths=0.5, fmt=".2f", ax=ax)

            # Passer l'objet figure à st.pyplot()
            st.pyplot(fig)

            st.write("On observe une corrélation positive entre la tranche d'âge 61_95 ans et la gravité de l'accident. De même, on observe une corrélation positive entre le nombre d'hommes impliqués et la gravité de l'accident.")
            
          

        if "Corrélation selon les conditions de la route" in variables_matrices_corrélation:
            @st.cache_data
            def get_filtered_data(df):
                to_drop = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]
                df_filtré = df.drop(df.columns[to_drop], axis=1)
                df_filtré['nbv'] = df['nbv'].replace(to_replace=['#ERREUR', '#VALEURMULTI'], value=np.nan)
                return df_filtré

            @st.cache_data
            def compute_correlation(df, selected_columns):
                # Calculer la matrice de corrélation pour les colonnes sélectionnées
                return df[selected_columns].corr()

            # Appliquer le pré-traitement
            df = get_filtered_data(df_total_final)

            # Colonnes sélectionnées pour la corrélation
            selected_columns3 = ['gravité_accident', 'nationale_departementale_communale', 'autoroute', 'autre_route',
                                'sens_unique', 'vma', 'bidirectionnel', 'route_seche', 'route_mouillee_enneigee',
                                'etat_route_autre', 'nbv', 'blessé_léger', 'blessé_hospitalisé', 'tué']

            
            # Calcul de la matrice de corrélation
            corr_matrix_subset3 = compute_correlation(df, selected_columns3)

            # Affichage de la matrice de corrélation avec Streamlit
            st.subheader("Matrice de corrélation selon les conditions de la route")

            # Créer un objet figure
            fig, ax = plt.subplots(figsize=(12, 8))

            # Tracer la carte de chaleur sur l'axe
            sns.heatmap(corr_matrix_subset3, annot=True, cmap="coolwarm", linewidths=0.5, fmt=".2f", ax=ax)

            # Afficher la figure dans Streamlit
            st.pyplot(fig)

            st.write("On observe une corrélation positive entre la vitesse maximale autorisée et la gravité de l'accident: plus la vitesse maximale autorisée augmente, plus la gravité de l'accident augmente.")
            st.write("On observe une corrélation positive forte entre la bidirectionnalité et la gravité")
            
            
        if "Corrélation selon les obstacles et types de chocs" in variables_matrices_corrélation:
            def get_filtered_data(df):
                to_drop = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]
                df_filtré = df.drop(df.columns[to_drop], axis=1)
                return df_filtré
            
            @st.cache_data
            def compute_correlation(df, selected_columns):
                # Calculer la matrice de corrélation pour les colonnes sélectionnées
                return df[selected_columns].corr()
            
            selected_columns2 = ['gravité_accident', 'obstacle_fixe', 'obstacle_mobile', 'aucun_choc', 'choc_AV', 'choc_AR', 
                     'choc_cote', 'choc_tonneaux', 'blessé_léger', 'blessé_hospitalisé', 'tué']

            df_filtré = get_filtered_data(df_total_final)
            # Calcul de la matrice de corrélation
            corr_matrix_subset2 = compute_correlation(df_filtré, selected_columns2)

            # Affichage de la matrice de corrélation avec Streamlit
            st.subheader("Matrice de corrélation selon les obstacles et types de chocs")

            # Créer un objet figure
            fig, ax = plt.subplots(figsize=(12, 8))

            # Tracer la carte de chaleur sur l'axe
            sns.heatmap(corr_matrix_subset2, annot=True, cmap="coolwarm", linewidths=0.5, fmt=".2f", ax=ax)

            # Afficher la figure dans Streamlit
            st.pyplot(fig)
            




elif st.session_state.page == "Modélisation":
    st.header("Modélisation")
    #"Evaluation modèles 3 classes"
    if Méthodologie_et_résultats_checkbox:
        tab1, tab2= st.tabs(["Méthodologie", " Synthèse des résultats"])

# Contenu du premier onglet
        with tab1:
            st.header("Méthodologie")
            paragraphe = """ L’objectif est de tester différents modèles, différentes configurations, dans le but d'obtenir les meilleures prédictions possible.
            " \
            Rappel: Notre variable gravité est séparée en trois catégories:
            - Blessé léger
            - Blessé hospitalisé
            - Tué" \
            "
            Quelle est la classe dont nous souhaitons optimiser la prédiction?
            Nous souhaitons prédire les gravité de l'accident afin de mobiliser les secours adéquats. Il nous importe donc de prédire au mieux la classe des blessés hospitalisés: en effet, pour les blessés légers, pas d'urgence de secours, et malheureusement, pour les tués non plus.
            " \
            "" \
            Comment évaluer la performance des modèles? Que cherchons nous à optimiser?
            Nous désirons que notre modèle prédise au mieux les vrais positifs pour avoir suffisamment de secours pour cette classe, et qu'il indique le moins de faux positifs pour ne pas mobiliser de secours inutilement.
            Notre but est donc d'optimiser le Recall (bonne prédiction des vrais positifs), et le F1_Score( peu de faux positifs)" \
           
            Quels modèles allons-nous tester?
            Il s'agit d'un problème de classification à 3 classes. Notons que ces classes sont très déséquilibrées, nous appliquerons donc oversampling et undersampling, et personnaliserons une métrique pondérée pour gérer au mieux ce déséquilibre.
            Nous testons donc différents modèles de classification, et différentes méthodes de gestion du déséquilibre des classes
            :
            Random Forest, LogisticRegression, XGBoost, KNN. 

            Nous partageons notre jeu de données en deux parties: X_train qui représente 80% des données, et X_test qui représente 20% des données.

             """
            
            st.write(paragraphe)

# Contenu du deuxième onglet
        with tab2:
            st.header("Synthèse des résultats")
            st.write('')

            st.subheader("Modèles 3 classes")
            modeles_selectionnes_3 = st.selectbox(
            "Sélectionnez un modèle :",
            ["Random Forest_3_classes", "Logistic Regression_3_classes", "KNN_3_classes", "XGBoost_3_classes"])
        
            if modeles_selectionnes_3 == "Random Forest_3_classes":
                image_path = r"C:\Users\macha\Desktop\Test_py\Random_Forest.jpg"
                st.image(image_path, use_column_width=True)

            elif modeles_selectionnes_3 == "Logistic Regression_3_classes":
                image_path = r"C:\Users\macha\Desktop\Test_py\Logistic_Reg.jpg"
                st.image(image_path, use_column_width=True)

            elif modeles_selectionnes_3 == "KNN_3_classes":
                image_path = r"C:\Users\macha\Desktop\Test_py\KNN.jpg"
                st.image(image_path, use_column_width=True)

            elif modeles_selectionnes_3 == "XGBoost_3_classes":
                image_path = r"C:\Users\macha\Desktop\Test_py\XGBOOST.jpg"
                st.image(image_path, use_column_width=True)
               

        
            st.subheader("Modèles 2 classes")
            modeles_selectionnes_2 = st.selectbox(
                "Sélectionnez un modèle :",["Random Forest_2_classes", "Logistic Regression_2_classes", "KNN_2_classes", "XGBoost_2_classes"])
        
            if modeles_selectionnes_2 == "Random Forest_2_classes":
                image_path = r"C:\Users\macha\Desktop\Test_py\2_classes_Random_Forest.jpg"
                st.image(image_path, use_column_width=True)
            elif modeles_selectionnes_2 == "Logistic Regression_2_classes":
                image_path = r"C:\Users\macha\Desktop\Test_py\2_classes_Logistic_Reg.jpg"
                st.image(image_path, use_column_width=True)
            elif modeles_selectionnes_2 == "KNN_2_classes":
                image_path = r"C:\Users\macha\Desktop\Test_py\2_classes_KNN.jpg"
                st.image(image_path, use_column_width=True)
            elif modeles_selectionnes_2 == "XGBoost_2_classes":
                image_path = r"C:\Users\macha\Desktop\Test_py\2_classes_XGBOOST.jpg"
                st.image(image_path, use_column_width=True)
    
    if Prédictions_checkbox:
        st.subheader('Prédictions')
        tab1, tab2= st.tabs(["Méthodologie", "Simulation"])

        with tab1:
            paragraphe = """ Nous imaginons que les véhicules futurs pourraient être équipés d’un boîtier, comparable à la boîte noire d’un avion, qui enregistrerait un nombre défini de paramètres (par connexion au GPS et interaction avec le conducteur) et déclencherait une alerte aux secours en cas d’accident.
            Les boitiers de chaque véhicule accidenté pourraient combiner leurs données puis appliquer le modèle de machine learning, et enfin envoyer au centre de secours le nombre prédit de blessés hospitalisés. 
            Il parait réaliste que le boitier dispose des paramètres suivants:
            'heure', 'agg', 'nbv', 'vma','homme', 'femme', '0-17','18-60', '61-95','PL', 'bus_car', 'VL_VU', '2roues_3roues_quad'.
            Nous reprenons alors les deux meilleurs modèles obtenus, Random Forest 3 classes et XGBoost 2 classes, et testons avec différentes variables et différents paramères pour s'apporcher au maximum des résultats précédemment obtenus 
            et présentés dans la partie précédente.
        
            """
            st.write(paragraphe)
            st.write("Pour le modèle Random Forest 3 classes, nous retenons ces variables: 'heure', 'agg', 'nbv', 'vma','homme', 'femme', '0-17','18-60', '61-95' et obtenons les scores suivants:")
            image_path= r"C:\Users\macha\Desktop\Test_py\Boitier_3_classes_Random_Forest.png"
            st.image(image_path, use_column_width=True)
            st.write("Pour le modèle XGBoost 2 classes, nous avons besoin de plus de variables pour un résultat pertinent: 'heure', 'agg', 'nbv', 'vma','homme', 'femme', '0-17','18-60', '61-95','PL', 'bus_car', 'VL_VU', '2roues_3roues_quad' et obtenons les scores suivants:")
            image_path= r"C:\Users\macha\Desktop\Test_py\Boitier_2_classes_XGBoost.png"
            st.image(image_path, use_column_width=True)
            st.write("Les modèles sont tout de même moins performants avec cette sélection de variables, mais il parait pertinent, pour la prédiction, de s'adapter à une situation concrète où les variables pourraient etre renseignées par un boitier intégré à la voiture.")

        with tab2:
            model_choice = st.selectbox("Choisissez un modèle :", [" ", "Random Forest 3 classes", "XGBoost 2 classes"])

            # Charger le bon modèle en fonction du choix
            if model_choice == "Random Forest 3 classes":
                    
                model = joblib.load("model_boitier_random_forest_3_classes.joblib")  # Remplace par ton chemin réel
                
                preprocessor_random = joblib.load("preprocessor_random.joblib")


                st.write("")
                st.write('Conditions')
                st.write("")
                # Widgets pour choisir les valeurs des features
                col1, col2, col3 = st.columns(3)

            
                with col1:
                    agg = st.selectbox("Localisation", ["Hors_agglomération", "En agglomération"])
                with col2:
                    atm = st.selectbox("Atmosphère", ['temps_normal', 'Temps_pluvieux', 'Temps_couvert', 'Autre'])
                with col3:
                    heure = st.time_input("Heure", value=time(12, 0))
                
                st.write("")
                st.write('Route')
                st.write("")

                col4,col5 = st.columns(2)
                
                with col4:
                    nbv= st.slider("Nombre de voies", min_value=0, max_value=10, value=0)
                with col5:
                    vma = st.selectbox("Vitesse maximale autorisée", [20, 30, 50, 60, 70, 80, 90,100, 110, 130])

                st.write("")
                st.write('Sexe')
                st.write("")

                col6,col7 = st.columns(2)
                
                with col6:
                    homme = st.slider("Nombre d'hommes", min_value=0, max_value=10, value=0)
                with col7:
                    femme = st.slider("Nombre de femmes", min_value=0, max_value=10, value=0)
                
                st.write("")
                st.write('Age')
                st.write("")
                
                col8, col9, col10 = st.columns(3)
                
                with col8:
                    inf_17 = st.slider("Nombre de personnes ayant 17 ans ou moins", min_value=0, max_value=10, value=0)
                with col9:
                    entre_18_60 = st.slider("Nombre de personnes entre 18 et 60 ans", min_value=0, max_value=10, value=0)
                with col10:
                    entre_61_95= st.slider("Nombre de personnes ayant plus de 61 ans", min_value=0, max_value=10, value=0)

            
                input_data_random = pd.DataFrame([{
                'agg': agg,
                'atm': atm,
                'heure': heure.hour + heure.minute / 60,  # Convertir l'heure en une valeur numérique
                'nbv': nbv,
                'vma': vma,
                'homme': homme,
                'femme': femme,
                '0-17': inf_17,
                '18-60': entre_18_60,
                '61-95': entre_61_95
            }])

                # Bouton pour la prédiction
                if st.button("⚡ Prédire"):
                    # Appliquer le ColumnTransformer
                    new_data_transformed_random = preprocessor_random.transform(input_data_random)

                    # Faire la prédiction avec le modèle
                    prediction_random = model.predict(new_data_transformed_random)

                    # Dictionnaire pour renommer les prédictions
                    st.write('Le blessé le plus grave lors de cet accident est:')
                    labels_random = {0: "un blessé_léger", 1: "Un blessé grave", 2: "Un tué"}
                    # La prédiction est un tableau, extraire la valeur
                    if isinstance(prediction_random, (list, np.ndarray)):  
                        prediction_random = prediction_random[0]

                    # Afficher le résultat
                    st.success(f"📌 La prédiction est : **{labels_random.get(prediction_random, 'Inconnu')}**")




            else:
                    
                model = joblib.load("model_boitier_XGBoost_2_classes.joblib")  # Remplace par ton chemin réel
                preprocessor = joblib.load("preprocessor.joblib")
                # Définition de la classe pour la transformation de l'heure
            
                

                st.write("")
                st.write('Conditions')
                st.write("")
                # Widgets pour choisir les valeurs des features
                col1, col2, col3 = st.columns(3)
                with col1:
                    agg = st.selectbox("Localisation", ["Hors_agglomération", "En agglomération"], key="agg_box")
                with col2:
                    atm = st.selectbox("Atmosphère", ['temps_normal', 'Temps_pluvieux', 'Temps_couvert', 'Autre'], key="atm_slider")
                with col3:
                    heure = st.time_input("Sélectionnez une heure", value=time(12, 0), key="heure_timer")

                st.write("")
                st.write('Route')
                st.write("")
                
                col4,col5 = st.columns(2)
                with col4:
                    nbv= st.slider("Nombre de voies", min_value=0, max_value=10, value=0, key="nbv_slider")
                with col5:
                    vma = st.selectbox("Vitesse maximale autorisée", [20, 30, 50, 60, 70, 80, 90,100, 110, 130],key="vma_box")

                st.write("")
                st.write('Sexe')
                st.write("")

                col6,col7 = st.columns(2)
                with col6:
                    homme = st.slider("Nombre d'hommes", min_value=0, max_value=10, value=0, key="homme_slider")
                with col7:
                    femme = st.slider("Nombre de femmes", min_value=0, max_value=10, value=0,key="femme_slider")
                
                st.write("")
                st.write('Type de véhicules')
                st.write("")
                
                col8,col9 = st.columns(2)
                with col8:
                    PL = st.slider("Nombre de poids lourds", min_value=0, max_value=10, value=0, key = 'PL_slider')
                with col9:
                    bus_car= st.slider("Nombre de bus/cars", min_value=0, max_value=10, value=0, key = 'bus_car_slider')
                
                col10,col11 = st.columns(2)
                with col10:
                    VL_VU= st.slider("Nombre de véhicules légers/ tilitaires", min_value=0, max_value=10, value=0, key ='VL_VU_slider')
                with col11:
                    deux_roues= st.slider("Nombre de deux roues", min_value=0, max_value=10, value=0, key = 'deux roues_slider')

                st.write("")
                st.write('Age')
                st.write("")

                col12, col13, col14 = st.columns(3)
                with col12:
                    inf_17 = st.slider("Nombre de personnes ayant 17 ans ou moins", min_value=0, max_value=10, value=0, key="inf_17_slider")
                with col13:
                    entre_18_60 = st.slider("Nombre de personnes entre 18 et 60 ans", min_value=0, max_value=10, value=0,key="entre18_60_slider")
                with col14:
                    entre_61_95= st.slider("Nombre de personnes ayant plus de 61 ans", min_value=0, max_value=10, value=0,key="entre_61_95_slider")

            
                input_data = pd.DataFrame([{
                'agg': agg,
                'atm': atm,
                'heure': heure.hour + heure.minute / 60,  # Convertir l'heure en une valeur numérique
                'nbv': nbv,
                'vma': vma,
                'homme': homme,
                'femme': femme,
                'PL': PL,
                'bus_car': bus_car,
                'VL_VU': VL_VU,
                '2roues_3roues_quad': deux_roues,
                '0-17': inf_17,
                '18-60': entre_18_60,
                '61-95': entre_61_95
            }])

                # Bouton pour la prédiction
                if st.button("⚡ Prédire", key= 'xgboost'):
                    new_data_transformed = preprocessor.transform(input_data) # Appliquer le ColumnTransformer
                
                    # Faire la prédiction avec le modèle
                    prediction = model.predict(new_data_transformed)
                    st.write('Le blessé le plus grave lors de cet accident est:')
                    labels = {0: "Non urgent (blessé ou tué)", 1: "Urgent (blessé grave)", 2: "Un tué"}

                # La prédiction est un tableau, extraire la valeur
                    if isinstance(prediction, (list, np.ndarray)):  
                        prediction = prediction[0]

                    # Afficher le résultat avec le label correspondant
                    st.success(f"📌 La prédiction est : **{labels.get(prediction, 'Inconnu')}**")

    elif st.session_state.page == "Conclusion":
        st.subheader('Conclusion')
        paragraphe = """Nous avons testé ici différents modèles et avons observé des résultats pouvant varier de manière très significative selon le type de modèle d’un part, mais également selon les paramètres choisis et les variables prises en compte d’autre part.

    Dans sa mise en application concrète, notre modèle pourrait avoir un véritable intérêt dans la gestion des accidents de la route par les secours par le biais du boitier.

    Dans cette optique, nos modèles nécessiteraient bien entendu une performance améliorée par rapport à nos résultats actuels, et différents axes d’améliorations pourraient être envisagés: 

    - Retravailler les données: en effet un modèle performant repose sur des données bien préparées et traitées en amont. Nous pourrions envisager d’utiliser certaines variables que nous avons mises de côté, et créer de nouvelles variables pour apporter une valeur ajoutée à notre dataset
    - Nous nous sommes concentrés sur les années 2019-2023, il pourrait être intéressant d'entrainer les modèles et d’étudier les résultats en incluant davantage d’années de données.
    - Mieux cibler les variables possible à enregistrer dans un boitier, tester différents modèles et paramètres avec ces variables uniquement, et trouver un modèle avec des performances de recall et F1-score bien plus élevées.
    Les modèles actuels prédisent la gravité la plus importante de l'accident, mais il serait important qu'il puisse prédire le nombre de blessés de chaque classe.

    Néanmoins, ce projet a été pour nous une première occasion de mettre en pratique les différentes compétences apprises au cours de notre formation, notamment l’étude, la compréhension et le travail du dataset, la mise en application du concept de machine learning et l’interprétation des résultats obtenus."""

        st.write(paragraphe)