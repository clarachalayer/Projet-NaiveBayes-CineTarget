#importation des librairies
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
import warnings
warnings.filterwarnings("ignore")


# config de la page 

st.set_page_config(
    page_title="CinéTarget — Ciblage Marketing",
    page_icon="🎬",
    layout="wide",
    initial_sidebar_state="expanded"
)


# Code Css pour définir le style de notre app

st.markdown("""
<style>
.stApp { background-color: #0f1117; }
section[data-testid="stSidebar"] { background-color: #1a1d27 !important; }

.stApp, .stApp p, .stApp li, .stApp label,
.stApp span, .stMarkdown, div[data-testid="stMarkdownContainer"] p {
    color: #e8eaf0 !important;
}

.kpi-card {
    background: linear-gradient(135deg, #1e2235, #252a3d);
    border: 1px solid #3a3f5c;
    border-radius: 14px;
    padding: 1.2rem 1rem;
    text-align: center;
    margin-bottom: 0.5rem;
}
.kpi-value { font-size: 2.2rem; font-weight: 900; color: #7c8dff; margin: 0; }
.kpi-label { font-size: 0.82rem; color: #9aa0b8; margin: 0; text-transform: uppercase; letter-spacing: 0.05em; }
.kpi-delta { font-size: 0.95rem; font-weight: 700; color: #4ade80; }

.sec-header {
    font-size: 1.25rem; font-weight: 800; color: #c5caff;
    border-left: 4px solid #7c8dff; padding-left: 0.75rem;
    margin: 2rem 0 0.8rem 0;
}

.info-box {
    background: #1a2035;
    border: 1px solid #2d3454;
    border-left: 4px solid #7c8dff;
    border-radius: 10px;
    padding: 1rem 1.2rem;
    color: #c8cde8 !important;
    margin: 0.6rem 0;
}
.info-box b { color: #a5b4ff !important; }

.warn-box {
    background: #1f1a10;
    border: 1px solid #4a3500;
    border-left: 4px solid #f59e0b;
    border-radius: 10px;
    padding: 1rem 1.2rem;
    color: #e8d5a0 !important;
    margin: 0.6rem 0;
}
.warn-box b { color: #fbbf24 !important; }

.success-box {
    background: #101f16;
    border: 1px solid #1a4a28;
    border-left: 4px solid #22c55e;
    border-radius: 10px;
    padding: 1rem 1.2rem;
    color: #a8dfb8 !important;
    margin: 0.6rem 0;
}
.success-box b { color: #4ade80 !important; }

.pred-card {
    border-radius: 16px;
    padding: 1.5rem;
    text-align: center;
    margin-bottom: 1rem;
}
.pred-genre { font-size: 2.4rem; font-weight: 900; margin: 0.3rem 0; }
.pred-conf { font-size: 1.1rem; color: #9aa0b8; }
.pred-model { font-size: 0.8rem; text-transform: uppercase; letter-spacing: 0.08em; color: #9aa0b8; }

.stTabs [data-baseweb="tab-list"] { gap: 6px; background: #1a1d27; border-radius: 10px; padding: 4px; }
.stTabs [data-baseweb="tab"] { border-radius: 8px; color: #9aa0b8 !important; }
.stTabs [aria-selected="true"] { background: #7c8dff22 !important; color: #c5caff !important; }

.js-plotly-plot { border-radius: 12px; }
.stDataFrame { border-radius: 10px; overflow: hidden; }

section[data-testid="stSidebar"] * { color: #c8cde8 !important; }
section[data-testid="stSidebar"] .stRadio label { color: #c8cde8 !important; }
</style>
""", unsafe_allow_html=True)


# définition des genres et du styles de chaque 

TOP_GENRES = ['Action', 'Horror', 'Comedy', 'Drama']

GENRE_COLORS = {
    'Action':  '#f5576c',
    'Horror':  '#8b5cf6',
    'Comedy':  '#f59e0b',
    'Drama':   '#3b82f6',
}

GENRE_EMOJI = {'Action': '💥', 'Horror': '👻', 'Comedy': '😂', 'Drama': '🎭'}

ORDINAL_MAP = {'No Rate': 0, 'Mild': 1, 'Moderate': 2, 'Severe': 3}

BASE_FEATURES = ['Date', 'Rate', 'Votes_clean', 'Duration', 'Type_enc',
                 'Certificate_enc', 'Nudity', 'Violence', 'Profanity',
                 'Alcohol', 'Frightening']
ENG_FEATURES  = ['content_intensity', 'rate_x_logvotes', 'is_recent', 'is_long']
ALL_FEATURES  = BASE_FEATURES + ENG_FEATURES

FEATURE_LABELS = {
    'Date':              'Année de sortie',
    'Rate':              'Note IMDB',
    'Votes_clean':       'Nombre de votes',
    'Duration':          'Durée (min)',
    'Type_enc':          'Type (Film/Série)',
    'Certificate_enc':   'Classification âge',
    'Nudity':            'Nudité',
    'Violence':          'Violence',
    'Profanity':         'Grossièreté',
    'Alcohol':           'Alcool / Drogues',
    'Frightening':       'Scènes effrayantes',
    'content_intensity': 'Intensité contenu (score)',
    'rate_x_logvotes':   'Note × log(Votes)',
    'is_recent':         'Sorti après 2015',
    'is_long':           'Durée ≥ 120 min',
}

MARKETING_ADVICE = {
    'Action': {
        'emoji': '💥',
        'audience': '18–34 ans, masculin dominant',
        'channels': 'YouTube Ads, Twitch, Instagram Reels, Cinéma',
        'timing': 'Été (juin–août) et fêtes (déc.)',
        'budget': 'Élevé — ROI fort sur le gros public',
        'color': '#f5576c',
    },
    'Horror': {
        'emoji': '👻',
        'audience': '18–29 ans, audiences nocturnes',
        'channels': 'TikTok viral, YouTube, Reddit, podcasts true crime',
        'timing': 'Octobre (Halloween), janvier',
        'budget': 'Modéré — fort multiplicateur viral',
        'color': '#8b5cf6',
    },
    'Comedy': {
        'emoji': '😂',
        'audience': '15–40 ans, tous publics',
        'channels': 'Réseaux sociaux, Netflix, memes, influenceurs',
        'timing': "Toute l'année — pic printemps et Noël",
        'budget': 'Moyen — fort engagement social',
        'color': '#f59e0b',
    },
    'Drama': {
        'emoji': '🎭',
        'audience': '25–54 ans, cultivé, urbain',
        'channels': 'Presse culturelle, Arte, Canal+, newsletters ciné',
        'timing': 'Automne (saison des festivals), janvier (awards)',
        'budget': 'Ciblé — qualité > quantité',
        'color': '#3b82f6',
    },
}

def hex_to_rgba(hex_color, alpha=0.2):
    h = hex_color.lstrip('#')
    r, g, b = int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16)
    return f'rgba({r},{g},{b},{alpha})'

# chargement de la pipeline du projet 

@st.cache_data
def load_and_train(filepath):
    df = pd.read_csv(filepath)

    for col in ['Rate', 'Duration', 'Date']:
        df[col] = pd.to_numeric(df[col], errors='coerce')
    df['Votes_clean'] = pd.to_numeric(
        df['Votes'].str.replace(',', '').str.replace('No Votes', '').str.strip(),
        errors='coerce'
    )

    for col in ['Nudity', 'Violence', 'Profanity', 'Alcohol', 'Frightening']:
        df[col] = df[col].map(ORDINAL_MAP)

    cert_enc = LabelEncoder()
    df['Certificate_enc'] = cert_enc.fit_transform(df['Certificate'].fillna('Unknown'))
    df['Type_enc'] = (df['Type'] == 'Film').astype(int)

    df['content_intensity'] = df[['Violence','Nudity','Frightening','Profanity','Alcohol']].fillna(0).sum(axis=1)
    df['rate_x_logvotes']   = df['Rate'].fillna(0) * np.log1p(df['Votes_clean'].fillna(0))
    df['is_recent']         = (df['Date'] >= 2015).astype(int)
    df['is_long']           = (df['Duration'] >= 120).astype(int)

    def assign_genre(g):
        if pd.isna(g): return None
        gs = [x.strip() for x in g.split(',')]
        for p in ['Horror', 'Action', 'Comedy', 'Drama']:
            if p in gs: return p
        return None

    df['target_genre'] = df['Genre'].apply(assign_genre)
    df = df.dropna(subset=['target_genre'])
    raw_counts = df['target_genre'].value_counts()

    min_n = raw_counts.min()
    df_bal = pd.concat([
        df[df['target_genre'] == g].sample(n=min_n, random_state=42)
        for g in TOP_GENRES
    ]).reset_index(drop=True)

    X = df_bal[ALL_FEATURES].copy()
    for col in ALL_FEATURES:
        X[col] = pd.to_numeric(X[col], errors='coerce')
    med = X.median()
    X = X.fillna(med)
    y = df_bal['target_genre']

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    scaler = MinMaxScaler()
    X_tr_sc = scaler.fit_transform(X_train)
    X_te_sc = scaler.transform(X_test)

    nb = GaussianNB()
    nb.fit(X_tr_sc, y_train)
    nb_pred  = nb.predict(X_te_sc)
    nb_proba = nb.predict_proba(X_te_sc)
    nb_acc   = accuracy_score(y_test, nb_pred)

    rf = RandomForestClassifier(n_estimators=200, random_state=42, n_jobs=-1)
    rf.fit(X_train, y_train)
    rf_pred  = rf.predict(X_test)
    rf_proba = rf.predict_proba(X_test)
    rf_acc   = accuracy_score(y_test, rf_pred)

    return {
        'df_raw': df, 'df_bal': df_bal,
        'X': X, 'y': y,
        'X_train': X_train, 'X_test': X_test, 'y_train': y_train, 'y_test': y_test,
        'X_tr_sc': X_tr_sc, 'X_te_sc': X_te_sc,
        'nb': nb, 'rf': rf,
        'nb_pred': nb_pred, 'nb_proba': nb_proba, 'nb_acc': nb_acc,
        'rf_pred': rf_pred, 'rf_proba': rf_proba, 'rf_acc': rf_acc,
        'scaler': scaler, 'cert_enc': cert_enc,
        'raw_counts': raw_counts, 'min_n': min_n,
    }


# Définition de la bar de navigation à gauche de chaque page

with st.sidebar:
    st.markdown("## CinéTarget")
    st.markdown("**Outil de ciblage marketing** — Prédiction automatique du genre de film")
    st.markdown("---")
    st.markdown("### Navigation")
    page = st.radio("", [
        "Vue d'ensemble marketing",
        "Données & Features",
        "Comparaison des modèles",
        "Cibler un film",
    ], label_visibility="collapsed")

try:
    D = load_and_train("imdb.csv")
except Exception as e:
    st.error(f"Erreur : {e}")
    st.info("Vérifiez que le fichier `imdb.csv` est bien dans le même dossier que ce script.")
    st.stop()

genres_used = sorted(D['df_bal']['target_genre'].unique())


# première page : vue d'ensemble marketing et problèmatique 

if page == "Vue d'ensemble marketing":

    st.markdown("# CinéTarget")
    st.markdown("### Outil de ciblage marketing automatisé pour l'industrie du film")
    st.markdown("---")

    # Problématique
    st.markdown('<p class="sec-header">Problématique</p>', unsafe_allow_html=True)
    st.markdown("""
    <div class="info-box">
    <b>Un défi concret pour l'industrie du film :</b><br><br>
    Comment promouvoir efficacement un film sans lire son synopsis ?
    Comment adapter automatiquement la stratégie marketing au genre — audience, canaux, timing — sur un portefeuille de dizaines de films, sans erreur de ciblage ni gaspillage budgétaire ?
    Notre réponse : un modèle de Machine Learning qui, à partir des seules caractéristiques numériques d'un film (note IMDB, durée, classification d'âge, intensité du contenu…), prédit automatiquement son genre et génère instantanément la stratégie marketing adaptée.<br><br>
    <b>Les conséquences d'un mauvais ciblage :</b><br>
    &nbsp;&nbsp;• Un film d'action promu sur des canaux Drama (presse culturelle, Arte) → audience trop restreinte, ROI faible.<br>
    &nbsp;&nbsp;• Un Horror sorti hors période Halloween → manque le pic de consommation naturel du genre.<br>
    &nbsp;&nbsp;• Un Comedy traité comme un Drama → ton trop sérieux, perte d'engagement sur les réseaux sociaux.<br><br>
    <b>Notre solution :</b> Un modèle de Machine Learning qui, à partir des seules caractéristiques
    <i>numériques</i> d'un film (note IMDB, durée, classification d'âge, intensité du contenu…),
    <b>prédit automatiquement son genre</b> et génère instantanément des recommandations marketing personnalisées.
    Aucune lecture du synopsis, aucune intervention humaine — le ciblage est produit en quelques millisecondes.
    </div>
    """, unsafe_allow_html=True)

    st.markdown("---")

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.markdown(f'<div class="kpi-card"><p class="kpi-value">{len(D["df_raw"]):,}</p><p class="kpi-label">Films analysés</p></div>', unsafe_allow_html=True)
    with col2:
        st.markdown(f'<div class="kpi-card"><p class="kpi-value">{len(D["df_bal"]):,}</p><p class="kpi-label">Dataset équilibré</p></div>', unsafe_allow_html=True)
    with col3:
        st.markdown(f'<div class="kpi-card"><p class="kpi-value">{D["nb_acc"]:.0%}</p><p class="kpi-label">Précision Naive Bayes</p></div>', unsafe_allow_html=True)
    with col4:
        st.markdown(f'<div class="kpi-card"><p class="kpi-value">{D["rf_acc"]:.0%}</p><p class="kpi-label">Précision Random Forest</p></div>', unsafe_allow_html=True)

    st.markdown("---")

    # ── Fiches marketing par genre
    st.markdown('<p class="sec-header">Stratégie marketing par genre</p>', unsafe_allow_html=True)
    st.markdown("*Ce tableau est directement exploitable par l'équipe marketing une fois le genre prédit.*")

    cols = st.columns(4)
    for i, genre in enumerate(TOP_GENRES):
        adv = MARKETING_ADVICE[genre]
        with cols[i]:
            st.markdown(f"""
            <div style="background: {adv['color']}18; border: 1.5px solid {adv['color']}55;
                        border-radius: 14px; padding: 1.1rem; min-height: 260px;">
                <div style="font-size:2rem; text-align:center;">{adv['emoji']}</div>
                <div style="font-size:1.1rem; font-weight:800; color:{adv['color']};
                            text-align:center; margin-bottom:0.8rem;">{genre}</div>
                <div style="font-size:0.78rem; color:#c8cde8; line-height:1.7;">
                    <b style="color:{adv['color']};">Audience</b><br>{adv['audience']}<br><br>
                    <b style="color:{adv['color']};">Canaux</b><br>{adv['channels']}<br><br>
                    <b style="color:{adv['color']};">Timing</b><br>{adv['timing']}<br><br>
                    <b style="color:{adv['color']};">Budget</b><br>{adv['budget']}
                </div>
            </div>
            """, unsafe_allow_html=True)


# 3 PAGE : DONNÉES & FEATURES 

elif page == "Données & Features":

    st.markdown("# Données & Features")
    st.markdown("---")

    # ── Section équilibrage
    st.markdown('<p class="sec-header">Équilibrage des données</p>', unsafe_allow_html=True)
    st.markdown("Un dataset déséquilibré entraîne un modèle biaisé — inutilisable pour le marketing.")

    col1, col2 = st.columns(2)
    raw_counts = D['raw_counts'][D['raw_counts'].index.isin(TOP_GENRES)].reset_index()
    raw_counts.columns = ['Genre', 'Films']
    bal_counts = D['df_bal']['target_genre'].value_counts().reset_index()
    bal_counts.columns = ['Genre', 'Films']

    with col1:
        st.markdown('<p class="sec-header">Avant équilibrage</p>', unsafe_allow_html=True)
        fig1 = px.bar(raw_counts, x='Genre', y='Films', color='Genre',
                      color_discrete_map=GENRE_COLORS, text='Films')
        fig1.update_traces(textposition='outside', textfont_color='white')
        fig1.update_layout(showlegend=False, plot_bgcolor='rgba(0,0,0,0)',
                           paper_bgcolor='rgba(0,0,0,0)', font_color='#c8cde8',
                           yaxis=dict(gridcolor='#2a2f45'))
        st.plotly_chart(fig1, use_container_width=True)
        st.markdown("""
        <div class="warn-box">
        <b>Problème :</b> Drama a ~2× plus de films que Horror.
        Sans équilibrage, le modèle dit <i>\"Drama\"</i> par défaut inutilisable pour cibler.
        </div>""", unsafe_allow_html=True)

    with col2:
        st.markdown('<p class="sec-header">Après équilibrage (undersampling)</p>', unsafe_allow_html=True)
        fig2 = px.bar(bal_counts, x='Genre', y='Films', color='Genre',
                      color_discrete_map=GENRE_COLORS, text='Films')
        fig2.update_traces(textposition='outside', textfont_color='white')
        fig2.update_layout(showlegend=False, plot_bgcolor='rgba(0,0,0,0)',
                           paper_bgcolor='rgba(0,0,0,0)', font_color='#c8cde8',
                           yaxis=dict(gridcolor='#2a2f45'))
        st.plotly_chart(fig2, use_container_width=True)
        st.markdown(f"""
        <div class="success-box">
        <b>Solution :</b> Undersampling aléatoire à <b>{D['min_n']} films/genre</b>.
        Le modèle apprend chaque genre à égalité — les probabilités prédites
        sont exploitables pour le ciblage marketing.
        </div>""", unsafe_allow_html=True)

    st.markdown("""
    <div class="info-box">
    <b>Sans équilibrage :</b> 80% des films se verront attribuer le genre "Drama" →
    toutes les campagnes ciblent la même audience de 25-54 ans → budget gaspillé.<br><br>
    <b>Avec équilibrage :</b> Le modèle discrimine correctement Action vs Horror vs Comedy →
    chaque film reçoit la bonne stratégie de ciblage → meilleur ROI publicitaire.
    </div>
    """, unsafe_allow_html=True)

    st.markdown("---")

    st.markdown('<p class="sec-header">Importance des features (Random Forest)</p>', unsafe_allow_html=True)
    fi = pd.Series(D['rf'].feature_importances_, index=ALL_FEATURES).sort_values(ascending=True)
    fig_fi = px.bar(
        x=fi.values, y=[FEATURE_LABELS[f] for f in fi.index],
        orientation='h', color=fi.values,
        color_continuous_scale=[[0, '#2a2f45'], [1, '#7c8dff']],
        text=[f'{v:.1%}' for v in fi.values],
        title="Importance des features pour la prédiction de genre"
    )
    fig_fi.update_traces(textposition='outside', textfont_color='white')
    fig_fi.update_layout(coloraxis_showscale=False, plot_bgcolor='rgba(0,0,0,0)',
                         paper_bgcolor='rgba(0,0,0,0)', font_color='#c8cde8',
                         xaxis=dict(gridcolor='#2a2f45'))
    st.plotly_chart(fig_fi, use_container_width=True)

    st.markdown("""
    <div class="info-box">
    <b>Lecture marketing :</b><br>
    • <b>Note × log(Votes)</b> : un film très vu et bien noté est probablement un blockbuster Action ou Comedy.<br>
    • <b>Intensité du contenu</b> : violence + nudité + peur élevées → Horror ou Action.<br>
    • <b>Classification âge</b> : R-rated → Horror/Thriller, PG → Comedy/Drama familial.<br>
    • <b>Durée</b> : films longs (&gt;120 min) → Drama ou Action épique.
    </div>
    """, unsafe_allow_html=True)



# ════════════════════════════════════════════════════
# PAGE : COMPARAISON DES MODÈLES (NB + RF côte à côte)
# ════════════════════════════════════════════════════
elif page == "Comparaison des modèles":

    st.markdown("# Comparaison des modèles")
    st.markdown("Naive Bayes et Random Forest évalués sur les mêmes données de test.")
    st.markdown("---")

    # ── KPIs comparatifs
    delta = D['rf_acc'] - D['nb_acc']
    c1, c2, c3 = st.columns(3)
    with c1:
        st.markdown(f'<div class="kpi-card"><p class="kpi-value">{D["nb_acc"]:.0%}</p><p class="kpi-label">Naive Bayes</p></div>', unsafe_allow_html=True)
    with c2:
        st.markdown(f'<div class="kpi-card"><p class="kpi-value">{D["rf_acc"]:.0%}</p><p class="kpi-label">Random Forest</p></div>', unsafe_allow_html=True)
    with c3:
        st.markdown(f'<div class="kpi-card"><p class="kpi-value kpi-delta">+{delta:.0%}</p><p class="kpi-label">Gain RF / NB</p></div>', unsafe_allow_html=True)

    st.markdown("---")

    # ── Matrices de confusion côte à côte
    st.markdown('<p class="sec-header">Matrices de confusion</p>', unsafe_allow_html=True)
    col1, col2 = st.columns(2)

    with col1:
        st.markdown("**Naive Bayes**")
        cm_nb = confusion_matrix(D['y_test'], D['nb_pred'], labels=genres_used)
        fig_cm_nb = px.imshow(cm_nb, x=genres_used, y=genres_used,
                              color_continuous_scale=[[0,'#1a1d27'],[1,'#7c8dff']],
                              text_auto=True, title="Naive Bayes — Matrice de confusion")
        fig_cm_nb.update_layout(xaxis_title="Prédit", yaxis_title="Réel",
                                paper_bgcolor='rgba(0,0,0,0)', font_color='#c8cde8')
        st.plotly_chart(fig_cm_nb, use_container_width=True)

    with col2:
        st.markdown("**Random Forest**")
        cm_rf = confusion_matrix(D['y_test'], D['rf_pred'], labels=genres_used)
        fig_cm_rf = px.imshow(cm_rf, x=genres_used, y=genres_used,
                              color_continuous_scale=[[0,'#1a1d27'],[1,'#22c55e']],
                              text_auto=True, title="Random Forest — Matrice de confusion")
        fig_cm_rf.update_layout(xaxis_title="Prédit", yaxis_title="Réel",
                                paper_bgcolor='rgba(0,0,0,0)', font_color='#c8cde8')
        st.plotly_chart(fig_cm_rf, use_container_width=True)

    # ── F1-Score comparé par genre
    st.markdown('<p class="sec-header">F1-Score par genre</p>', unsafe_allow_html=True)
    nb_rep = classification_report(D['y_test'], D['nb_pred'], output_dict=True)
    rf_rep = classification_report(D['y_test'], D['rf_pred'], output_dict=True)
    rows = []
    for g in genres_used:
        rows.append({'Genre': g, 'Modèle': 'Naive Bayes',    'F1': nb_rep.get(g, {}).get('f1-score', 0)})
        rows.append({'Genre': g, 'Modèle': 'Random Forest',  'F1': rf_rep.get(g, {}).get('f1-score', 0)})
    fig_f1 = px.bar(pd.DataFrame(rows), x='Genre', y='F1', color='Modèle', barmode='group',
                    color_discrete_map={'Naive Bayes': '#7c8dff', 'Random Forest': '#22c55e'},
                    title="F1-Score par genre — comparaison directe", text_auto='.0%')
    fig_f1.update_layout(yaxis_range=[0, 1], plot_bgcolor='rgba(0,0,0,0)',
                         paper_bgcolor='rgba(0,0,0,0)', font_color='#c8cde8',
                         yaxis=dict(gridcolor='#2a2f45'))
    st.plotly_chart(fig_f1, use_container_width=True)



# ════════════════════════════════════════════════════
# PAGE : CIBLER UN FILM
# ════════════════════════════════════════════════════
elif page == "Cibler un film":

    st.markdown("# Cibler un film")
    st.markdown("Renseignez les caractéristiques du film pour obtenir le genre prédit.")

    with st.form("film_form"):
        col1, col2, col3 = st.columns(3)

        with col1:
            st.markdown("**Informations générales**")
            year     = st.slider("Année de sortie", 1990, 2025, 2022)
            rate     = st.slider("Note IMDB estimée", 1.0, 10.0, 6.8, 0.1)
            votes    = st.number_input("Nombre de votes estimé", 0, 5_000_000, 80_000, 5_000)
            duration = st.slider("Durée (minutes)", 30, 250, 105)
            ftype    = st.radio("Type", ["Film", "Série"], horizontal=True)

        with col2:
            st.markdown("**Classification**")
            all_certs = list(D['cert_enc'].classes_)
            cert = st.selectbox("Classification âge", all_certs)
            st.markdown("**Contenu**")
            nudity    = st.select_slider("Nudité",      ['No Rate','Mild','Moderate','Severe'], 'Mild')
            violence  = st.select_slider("Violence",    ['No Rate','Mild','Moderate','Severe'], 'Moderate')
            profanity = st.select_slider("Grossièreté", ['No Rate','Mild','Moderate','Severe'], 'Mild')

        with col3:
            st.markdown("**Contenu (suite)**")
            alcohol     = st.select_slider("Alcool/Drogues",     ['No Rate','Mild','Moderate','Severe'], 'Mild')
            frightening = st.select_slider("Scènes effrayantes", ['No Rate','Mild','Moderate','Severe'], 'Mild')
            st.markdown("&nbsp;")
            st.markdown("&nbsp;")
            submit = st.form_submit_button("Analyser ce film", type="primary", use_container_width=True)

    if submit:
        cert_enc_val = D['cert_enc'].transform([cert])[0] if cert in D['cert_enc'].classes_ else 0
        n_val   = ORDINAL_MAP[nudity]
        v_val   = ORDINAL_MAP[violence]
        p_val   = ORDINAL_MAP[profanity]
        a_val   = ORDINAL_MAP[alcohol]
        f_val   = ORDINAL_MAP[frightening]
        ci_val  = n_val + v_val + f_val + p_val + a_val
        rlv_val = rate * np.log1p(votes)

        x_raw = np.array([[year, rate, votes, duration, 1 if ftype == 'Film' else 0,
                           cert_enc_val, n_val, v_val, p_val, a_val, f_val,
                           ci_val, rlv_val, int(year >= 2015), int(duration >= 120)]])
        x_sc = D['scaler'].transform(x_raw)

        nb_probas  = D['nb'].predict_proba(x_sc)[0]
        rf_probas  = D['rf'].predict_proba(x_raw)[0]
        nb_classes = D['nb'].classes_
        rf_classes = D['rf'].classes_

        nb_genre = nb_classes[np.argmax(nb_probas)]
        rf_genre = rf_classes[np.argmax(rf_probas)]
        nb_conf  = np.max(nb_probas)
        rf_conf  = np.max(rf_probas)

        st.markdown("---")
        st.markdown("## Résultats de prédiction")

        col1, col2 = st.columns(2)
        for col, genre, conf, model_name in [
            (col1, nb_genre, nb_conf, "Naive Bayes"),
            (col2, rf_genre, rf_conf, "Random Forest"),
        ]:
            color = GENRE_COLORS.get(genre, '#7c8dff')
            emoji = GENRE_EMOJI.get(genre, '🎬')
            with col:
                st.markdown(f"""
                <div class="pred-card" style="background:{color}18; border:2px solid {color}66;">
                    <div class="pred-model">{model_name}</div>
                    <div style="font-size:3rem;">{emoji}</div>
                    <div class="pred-genre" style="color:{color};">{genre}</div>
                    <div class="pred-conf">Confiance : <b style="color:{color};">{conf:.1%}</b></div>
                </div>
                """, unsafe_allow_html=True)

        # Accord/désaccord
        if nb_genre == rf_genre:
            st.markdown(f"""
            <div class="success-box">
            <b>Les deux modèles s'accordent sur {GENRE_EMOJI.get(rf_genre,'')} {rf_genre}</b> —
            Confiance élevée. Vous pouvez lancer la campagne marketing.
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown(f"""
            <div class="warn-box">
            <b>Désaccord entre les modèles</b> : NB prédit {nb_genre}, RF prédit {rf_genre}.<br>
            Fiez-vous à <b>Random Forest</b> (plus précis), mais une vérification humaine est recommandée
            avant de lancer la campagne.
            </div>
            """, unsafe_allow_html=True)

        # Recommandation marketing basée sur le genre RF
        final = rf_genre
        adv = MARKETING_ADVICE[final]
        color = adv['color']

        st.markdown("---")
        st.markdown(f"## Recommandation marketing — {GENRE_EMOJI[final]} {final}")

        c1, c2, c3, c4 = st.columns(4)
        for col, icon, label, val in [
            (c1, '👥', 'Audience cible',    adv['audience']),
            (c2, '📣', 'Canaux prioritaires', adv['channels']),
            (c3, '📅', 'Meilleure période',  adv['timing']),
            (c4, '💰', 'Budget',             adv['budget']),
        ]:
            with col:
                st.markdown(f"""
                <div style="background:{color}15; border:1px solid {color}44; border-radius:12px;
                            padding:1rem; text-align:center; min-height:120px;">
                    <div style="font-size:1.5rem;">{icon}</div>
                    <div style="font-size:0.75rem; color:{color}; font-weight:700; text-transform:uppercase;
                                letter-spacing:0.05em; margin:0.3rem 0;">{label}</div>
                    <div style="font-size:0.85rem; color:#c8cde8;">{val}</div>
                </div>
                """, unsafe_allow_html=True)

# ════════════════════════════════════════════════════
# FOOTER
# ════════════════════════════════════════════════════
st.markdown("---")
st.markdown(
    "<div style='text-align:center; color:#4a5080; font-size:0.8rem;'>"
    "CinéTarget — Outil de ciblage marketing par prédiction de genre | "
    "Naive Bayes + Random Forest | Dataset IMDB"
    "</div>", unsafe_allow_html=True
)