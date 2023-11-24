import streamlit as st
import pandas as pd
import seaborn as sns
from sklearn.neighbors import NearestNeighbors
import matplotlib.pyplot as plt

# Page Présentation
def page_presentation():
    st.title("Présentation du projet")
    st.markdown("*Projet réalisé par: Julie, Mireille, Olmira, Maxime.* Étudiants à la Wild Code School")
    st.header("Sujet du Projet")
    st.write("Vous êtes un Data Analyst freelance. Un cinéma en perte de vitesse situé dans la Creuse vous contacte. "
             "Il a décidé de passer le cap du digital en créant un site Internet taillé pour les locaux. "
             "Pour aller encore plus loin, il vous demande de créer un moteur de recommandations de films qui à terme, "
             "enverra des notifications aux clients via Internet.")
    
    st.write("Pour l’instant, aucun client n’a renseigné ses préférences, vous êtes dans une situation de cold start. "
             "Mais heureusement, le client vous donne une base de données de films basée sur la plateforme IMDb.")
    
    st.write("Vous allez commencer par proposer une analyse complète de la base de données (Quels sont les acteurs les plus présents ? "
             "À quelle période ? La durée moyenne des films s’allonge ou se raccourcit avec les années ? Les acteurs de série sont-ils "
             "les mêmes qu’au cinéma ? Les acteurs ont en moyenne quel âge ? Quels sont les films les mieux notés ? Partagent-ils des caractéristiques "
             "communes ? etc…)")
    
    st.write("Suite à une première analyse, vous pouvez décider de spécialiser votre cinéma, par exemple sur la « période années 90 », ou alors sur « les films d’action et d’aventure », "
             "afin d'affiner votre exploration.")
    
    st.write("Après cette étape analytique, sur la fin du projet, vous utiliserez des algorithmes de machine learning pour recommander des films en fonction de films "
             "qui ont été appréciés par le spectateur.")
    
    st.write("Le client vous fournit également une base de données complémentaires venant de TMDB, contenant des données sur les pays des boîtes de production, "
             "le budget, les recettes et également un chemin vers les posters des films. Il vous est demandé de récupérer les images des films pour les afficher "
             "dans votre interface de recommandation.")
    
    st.write("Attention ! L’objectif n’est pas de diffuser dans le cinéma les films recommandés. L’objectif final est d’avoir une application avec d’une part des KPI et d’autre "
             "part le système de recommandation avec une zone de saisie de nom de film pour l’utilisateur. Cette application sera mise à disposition des clients du cinéma afin "
             "de leur proposer un service supplémentaire, en ligne, en plus du cinéma classique.")

# Page Recommandation
def page_recommandation():
    st.title("Recommandation")
    st.header("Objectif de la Recommandation")
    st.write("Bienvenue dans la section de recommandation de films ! Cette fonctionnalité permet aux utilisateurs de découvrir "
             "des films en fonction de leurs préférences. Entrez simplement le genre de film qui vous intéresse, et notre système "
             "vous fournira des recommandations basées sur une analyse approfondie de la base de données de films IMDb. Profitez de "
             "la diversité des recommandations et trouvez des films qui correspondent à vos goûts cinématographiques !")
    
    # Charger les données des films
    chemin_fichier_films = 'https://drive.google.com/file/d/16O94vNDUFSCSW9VZmqOtxlZjfDd0tjVP/view?usp=sharing'
    df_movies = pd.read_csv(chemin_fichier_films, sep='\t')

    # Remplacer les valeurs manquantes dans la colonne 'genres'
    df_movies['genres'] = df_movies['genres'].fillna('')

    # Sélectionner les colonnes numériques
    X_movies = df_movies[['averageRating', 'numVotes', 'startYear']]

    # Entraîner le modèle k-NN
    model_knn_movies = NearestNeighbors(n_neighbors=5)
    model_knn_movies.fit(X_movies)

    # Paramètres pour le calcul du score pondéré 'C' est la note moyenne de tous les films, 'm' est le paramètre ajustable 
    C = df_movies['averageRating'].mean()
    m = df_movies['numVotes'].quantile(0.90)

    # Saisie de l'utilisateur pour le genre
    genre_entre_par_utilisateur = st.text_input("Entrez le genre de films que vous recherchez:", "Drama")

    # Bouton pour lancer la recommandation
    if st.button("Obtenir des recommandations"):
        # Faire des prédictions pour le genre donné
        df_genre = df_movies[df_movies['genres'].str.lower().str.contains(genre_entre_par_utilisateur.lower())].copy()
        df_genre.dropna(subset=['averageRating', 'numVotes'], inplace=True)

        if not df_genre.empty:
            # Calculer le score pondéré 
            df_genre['weightedScore'] = df_genre.apply(lambda row: (row['numVotes'] / (row['numVotes'] + m)) * row['averageRating'] + (m / (row['numVotes'] + m)) * C, axis=1)

            # Trier les films par score pondéré en ordre décroissant
            df_genre = df_genre.sort_values(by='weightedScore', ascending=False)

            # Afficher les recommandations avec un format arrondi
            st.write(f"Recommandations de films par genre('{genre_entre_par_utilisateur}'):")
            for idx, row in df_genre.head(5).iterrows():
                st.write(f"{row['primaryTitle']} : Note: {round(row['weightedScore'], 2)}")
        else:
            st.write(f"Aucun film trouvé pour le genre '{genre_entre_par_utilisateur}'.")
    st.header("Le Code en Détail")
    st.write("Ci-dessous, vous trouverez une explication détaillée du code utilisé pour générer les recommandations de films. "
         "Le système de recommandation se base sur un fichier df_movies qui, au préalable, à été nettoyer selectionner les critères suivants: les films à partir des années 1990, en excluant ceux qui ont moins de 1000 votes. "
         "La catégorie Adult à été supprimer."
         "Le système de recommandation à été conçu en sorte de pouvoir aussi bien distinguer les films avec un genre unique, comme mutligenre "
         "Nous avons choisi cette approche pour nous concentrer sur des films plus récents et populaires. Le score pondéré est calculé "
         "pour chaque film en tenant compte de sa note moyenne, du nombre de votes, et en utilisant le paramètre ajustable 'm' pour "
         "assurer une certaine robustesse aux films avec un nombre limité de votes. Cette approche vise à recommander des films qui sont "
         "à la fois bien notés et populaires.")  
    st.code("""
    # Chargement des données des films "Tank, charge le programme de saut."-Matrix
    chemin_fichier_films = 'C:\\Users\\maxim\\OneDrive\\Bureau\\PROJET 2 NOTE PAR GENRE\\filtered_data.csv'
    df_movies = pd.read_csv(chemin_fichier_films)

    # Remplacer les valeurs manquantes dans la colonne 'genres' "Quel genre d'abstrait? plutôt braque, vasarely?"-Les 3 frères
    df_movies['genres'] = df_movies['genres'].fillna('')

    # Sélection des colonnes numériques "011101110110000101110010"-War cant of Mars
    X_movies = df_movies[['averageRating', 'numVotes', 'startYear']]

    # Entraîner le modèle k-NN "May the force be with you"- Star wars
    model_knn_movies = NearestNeighbors(n_neighbors=5)
    model_knn_movies.fit(X_movies)

    # Paramètres pour le calcul du score pondéré 'C' est la note moyenne de tous les films, 'm' est le paramètre ajustable 
    C = df_movies['averageRating'].mean()
    m = df_movies['numVotes'].quantile(0.90)
    #Put that cookie down! Now! -La course au jouet.

    # Fonction de recommandation avec score pondéré. "Vers l'infini et l'au delà" -Toys Story
    def recommander_films_par_genre(genre_entre_par_utilisateur):
        # Faire des prédictions pour le genre donné
        df_genre = df_movies[df_movies['genres'].str.lower().str.contains(genre_entre_par_utilisateur.lower())].copy()
        df_genre.dropna(subset=['averageRating', 'numVotes'], inplace=True)
        
        if not df_genre.empty:
            # Calcule du score pondéré 
            df_genre['weightedScore'] = df_genre.apply(lambda row: (row['numVotes'] / (row['numVotes'] + m)) * row['averageRating'] + (m / (row['numVotes'] + m)) * C, axis=1)
            
            # Trier les films par score pondéré en ordre décroissant (même si les croissants c'est meilleur.)
            df_genre = df_genre.sort_values(by='weightedScore', ascending=False)
            
            # Afficher les recommandations avec un format arrondi "à une vache prêt hein c'est pas une science exacte."- Kaamelot
            print(f"Recommandations de films par genre('{genre_entre_par_utilisateur}'):")
            for idx, row in df_genre.head(5).iterrows():
                print(f"{row['primaryTitle']} : Note: {round(row['weightedScore'], 2)}")
        else:
            print(f"Aucun film trouvé pour le genre '{genre_entre_par_utilisateur}'.")
    
    # Exemple d'utilisation "Why so serious?"- Batman The Dark Knight.
    recommander_films_par_genre('Drama')
    """)
    
    # Points positifs
    st.header("Points positifs")
    st.write("- Le modèle utilise une approche k-NN pour les recommandations, ce qui peut fournir des résultats pertinents.")
    st.write("- Le score pondéré prend en compte à la fois la note moyenne des films et le nombre de votes, offrant ainsi une recommandation plus équilibrée.")
    st.write("- La sélection des films se base sur des critères spécifiques, tels que les années 1990 et un nombre minimum de votes, pour des recommandations plus actuelles et populaires.")

    # Points négatifs
    st.header("Points négatifs")
    st.write("- Le modèle k-NN peut ne pas être optimal pour des jeux de données très larges et peut être sensible à la présence de valeurs aberrantes.")
    st.write("- Les paramètres tels que la période des années 1990 et le seuil de votes sont définis de manière statique, ce qui peut ne pas convenir à tous les utilisateurs.")
    
    # Idées futures
    st.header("Idées futures")
    st.write("- Permettre à l'utilisateur de personnaliser davantage ses préférences et affiner les recommandations.")
    st.write("- Explorer d'autres algorithmes de recommandation, tels que les méthodes de factorisation de matrices, pour comparer les performances.")
    st.write("- Intégrer des informations supplémentaires, telles que les genres spécifiques, pour améliorer la pertinence des recommandations.")
# Page Analyse
def page_analyse():
    st.title("Analyse")
    st.write("L'analyse ci-dessous met en lumière un KPI intéressant pour notre client. En se concentrant sur les genres de films "
             "les mieux notés, le cinéma peut mettre en avant ces catégories pour attirer un public plus large. Cette analyse peut "
             "aider le client à prendre des décisions éclairées sur la programmation et la mise en avant de certains types de films.")
    # Charger les fichiers cleaned
    url = "https://drive.google.com/file/d/18JYuj5VawGilFgvNjBfad5bo-j_6tQRi/view?usp=sharing"
    title_basics = pd.read_csv(url, sep='\t')
    lien = "https://drive.google.com/file/d/1HPYot1X3aY-iFOJpKkRHlZE-zL2hFKT7/view?usp=sharing"
    title_ratings = pd.read_csv(lien, sep='\t')

    # Convertir la colonne 'startYear' en numérique
    title_basics['startYear'] = pd.to_numeric(title_basics['startYear'], errors='coerce')

    # Filtrer les données dans chaque fichier
    filtered_basics = title_basics[(title_basics['titleType'] == 'movie') & 
                                    (title_basics['startYear'] >= 1990) & 
                                    (~title_basics['genres'].str.contains('Adult', na=False)) &
                                    (title_basics['genres'] != '\\N')]

    filtered_ratings = title_ratings[title_ratings['tconst'].isin(filtered_basics['tconst'])]

    # Fusionner les données
    merged_data = pd.merge(filtered_basics, filtered_ratings, on='tconst')

    # Séparer les genres multiples
    merged_data['genres'] = merged_data['genres'].str.split(',')

    # Appliquer explode pour convertir les listes de genres en plusieurs lignes
    merged_data_exploded = merged_data.explode('genres')

    # Sélectionner les N catégories de genre les plus fréquentes
    top_genres = merged_data_exploded['genres'].value_counts().head(10).index
    filtered_data_top_genres = merged_data_exploded[merged_data_exploded['genres'].isin(top_genres)]

    # Visualiser la distribution des notes par genre
    fig, ax = plt.subplots(figsize=(14, 8))
    sns.boxplot(x='genres', y='averageRating', data=filtered_data_top_genres, palette='viridis', ax=ax)
    plt.title('Distribution des notes par genre (Top 10 catégories)')
    plt.xlabel('Genre')
    plt.ylabel('Note moyenne')
    plt.xticks(rotation=45, ha='right')
    st.pyplot(fig)

# Barre latérale avec les onglets
page = st.sidebar.radio("Navigation", ["Présentation", "Recommandation", "Analyse"])

# Affiche la page sélectionnée
if page == "Présentation":
    page_presentation()
elif page == "Recommandation":
    page_recommandation()
elif page == "Analyse":
    page_analyse()

    st.header("Résumé")
    st.write("Le code ci-dessus utilise une boîte à moustaches (boxplot) pour visualiser la distribution des notes moyennes "
             "des dix catégories de genre les plus fréquentes. Cette visualisation permet d'identifier les genres de films qui "
             "ont tendance à recevoir de meilleures notes. Cette information peut guider le client dans la prise de décisions "
             "stratégiques pour la programmation du cinéma.")
    st.write("- **Médiane (ligne au milieu de la boîte) :** La médiane représente la valeur centrale de la distribution des notes. "
         "Elle permet d'obtenir une mesure de tendance centrale robuste aux valeurs extrêmes.")

    st.write("- **Quartiles (limites de la boîte) :** Les quartiles délimitent la moitié inférieure (Q1) et la moitié supérieure (Q3) "
         "de la distribution. Ils sont utiles pour comprendre la dispersion des notes dans chaque catégorie de genre.")

    st.write("- **Whiskers (moustaches) :** Les moustaches s'étendent jusqu'aux observations qui ne sont pas considérées comme des "
         "outliers. Elles fournissent une indication de la dispersion des notes tout en identifiant les valeurs atypiques.")

    st.write("- **Outliers (points en dehors des moustaches) :** Les points en dehors des moustaches représentent des observations "
         "considérées comme des valeurs extrêmes. Ils sont importants à identifier car ils peuvent indiquer des films exceptionnels "
         "dans une catégorie.")

    st.write("Cette visualisation offre une compréhension approfondie de la distribution des notes moyennes dans différentes catégories "
         "de films, en mettant en évidence les caractéristiques centrales, la dispersion et la présence d'outliers.")
