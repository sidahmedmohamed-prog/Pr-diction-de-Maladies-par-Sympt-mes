import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os

# ── Configuration de la page ──────────────────────────────────────────────────
st.set_page_config(
    page_title="Prédiction de Maladies",
    page_icon="🩺",
    layout="wide"
)

# ── Style CSS personnalisé ────────────────────────────────────────────────────
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;600;700&display=swap');

    html, body, [class*="css"] { font-family: 'Inter', sans-serif; }

    .header-box {
        background: linear-gradient(135deg, #1a1a2e 0%, #16213e 50%, #0f3460 100%);
        border-radius: 16px;
        padding: 2rem 2.5rem;
        margin-bottom: 2rem;
        border-left: 5px solid #e94560;
    }
    .header-box h1 { color: #ffffff; font-size: 2rem; margin: 0 0 0.5rem 0; }
    .header-box p  { color: #a0aec0; margin: 0; font-size: 0.95rem; }

    .result-box {
        background: #0d1117;
        border: 1px solid #238636;
        border-radius: 12px;
        padding: 1.5rem;
        font-family: monospace;
        font-size: 1rem;
        color: #3fb950;
        white-space: pre-wrap;
        margin-top: 1rem;
    }
    .warning-box {
        background: #1c1600;
        border: 1px solid #d29922;
        border-radius: 12px;
        padding: 1rem 1.5rem;
        color: #e3b341;
        margin-top: 1rem;
    }
    .stat-card {
        background: #161b22;
        border: 1px solid #30363d;
        border-radius: 12px;
        padding: 1.2rem;
        text-align: center;
    }
    .stat-card h2 { color: #e94560; font-size: 2rem; margin: 0; }
    .stat-card p  { color: #8b949e; margin: 0.3rem 0 0 0; font-size: 0.85rem; }

    div[data-testid="stTabs"] button {
        font-size: 1rem;
        font-weight: 600;
    }
    .stButton > button {
        border-radius: 8px;
        font-weight: 600;
        padding: 0.5rem 1.5rem;
    }
    footer { visibility: hidden; }
</style>
""", unsafe_allow_html=True)

# ── Chargement des modèles ────────────────────────────────────────────────────
@st.cache_resource
def charger_modeles():
    model         = joblib.load('model_rf_tuned.pkl')
    scaler        = joblib.load('scaler.pkl')
    label_encoder = joblib.load('label_encoder.pkl')
    symptom_cols  = joblib.load('symptom_cols.pkl')
    return model, scaler, label_encoder, symptom_cols

try:
    model, scaler, label_encoder, symptom_cols = charger_modeles()
    modeles_ok = True
except Exception as e:
    modeles_ok = False
    st.error(f"❌ Erreur de chargement des modèles : {e}")

# ── En-tête ───────────────────────────────────────────────────────────────────
st.markdown("""
<div class="header-box">
    <h1>🩺 Prédiction de Maladies par Symptômes</h1>
    <p>Projet Machine Learning — Classification multi-classe de <strong style="color:#e94560">42 maladies</strong> &nbsp;|&nbsp; Modèle : Random Forest optimisé</p>
</div>
""", unsafe_allow_html=True)

# ── Stats rapides ─────────────────────────────────────────────────────────────
if modeles_ok:
    c1, c2, c3 = st.columns(3)
    with c1:
        st.markdown('<div class="stat-card"><h2>42</h2><p>Maladies détectables</p></div>', unsafe_allow_html=True)
    with c2:
        st.markdown(f'<div class="stat-card"><h2>{len(symptom_cols)}</h2><p>Symptômes disponibles</p></div>', unsafe_allow_html=True)
    with c3:
        st.markdown('<div class="stat-card"><h2>RF</h2><p>Random Forest optimisé</p></div>', unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

# ── Onglets ───────────────────────────────────────────────────────────────────
tab1, tab2 = st.tabs(["🧑 Test individuel", "📂 Test par fichier CSV / Excel"])

# ════════════════════════════════════════════════════════════════════════════════
# ONGLET 1 — Prédiction individuelle
# ════════════════════════════════════════════════════════════════════════════════
with tab1:
    st.markdown("### Sélectionnez les symptômes du patient")
    st.caption("Cochez tous les symptômes observés chez le patient.")

    if modeles_ok:
        # Recherche de symptômes
        recherche = st.text_input("🔍 Rechercher un symptôme", placeholder="Tapez pour filtrer...")
        symptomes_filtres = [s for s in symptom_cols if recherche.lower() in s.lower()] if recherche else symptom_cols

        # Sélection multiple avec multiselect
        symptomes_selectionnes = st.multiselect(
            "Symptômes présents",
            options=symptom_cols,
            default=[],
            placeholder="Choisissez les symptômes..."
        )

        # Afficher les symptômes filtrés si recherche active
        if recherche and symptomes_filtres:
            st.info(f"💡 Symptômes correspondants : **{', '.join(symptomes_filtres[:10])}**{'...' if len(symptomes_filtres) > 10 else ''}")

        col1, col2 = st.columns([1, 3])
        with col1:
            reset = st.button("🔄 Réinitialiser", use_container_width=True)
        with col2:
            predire = st.button("🔍 Prédire la maladie", type="primary", use_container_width=True)

        if reset:
            st.rerun()

        if predire:
            if not symptomes_selectionnes:
                st.markdown('<div class="warning-box">⚠️ Veuillez sélectionner au moins un symptôme.</div>', unsafe_allow_html=True)
            else:
                with st.spinner("Analyse en cours..."):
                    vecteur = np.zeros(len(symptom_cols))
                    for s in symptomes_selectionnes:
                        if s in symptom_cols:
                            vecteur[symptom_cols.index(s)] = 1

                    X     = scaler.transform([vecteur])
                    pred  = model.predict(X)[0]
                    maladie = label_encoder.inverse_transform([pred])[0]
                    proba = model.predict_proba(X)[0]
                    top3_idx = np.argsort(proba)[::-1][:3]

                    texte  = f"✅ Maladie prédite : {maladie}\n\n"
                    texte += "📊 Top 3 probabilités :\n"
                    for i, idx in enumerate(top3_idx, 1):
                        nom = label_encoder.classes_[idx]
                        pct = proba[idx] * 100
                        texte += f"  {i}. {nom} : {pct:.2f}%\n"
                    texte += f"\n🔢 Symptômes sélectionnés : {len(symptomes_selectionnes)}"

                    st.markdown(f'<div class="result-box">{texte}</div>', unsafe_allow_html=True)

                    # Barre de confiance
                    st.markdown("<br>", unsafe_allow_html=True)
                    top_conf = proba[np.argsort(proba)[::-1][0]] * 100
                    st.metric("Confiance du modèle", f"{top_conf:.1f}%")
                    st.progress(int(top_conf))

# ════════════════════════════════════════════════════════════════════════════════
# ONGLET 2 — Prédiction par fichier
# ════════════════════════════════════════════════════════════════════════════════
with tab2:
    st.markdown("### Importez un fichier pour prédire plusieurs patients")
    st.caption("Le fichier doit contenir **une colonne par symptôme** (valeurs 0 ou 1).")

    if modeles_ok:
        fichier = st.file_uploader(
            "📎 Importer un fichier CSV ou Excel",
            type=["csv", "xlsx", "xls"]
        )

        if fichier is not None:
            try:
                if fichier.name.endswith(".csv"):
                    df = pd.read_csv(fichier)
                else:
                    df = pd.read_excel(fichier)

                st.success(f"✅ Fichier chargé : **{fichier.name}** — {len(df)} lignes, {len(df.columns)} colonnes")
                st.dataframe(df.head(5), use_container_width=True)

                colonnes_manquantes = [c for c in symptom_cols if c not in df.columns]

                if colonnes_manquantes:
                    apercu = ', '.join(colonnes_manquantes[:5])
                    st.error(f"❌ Colonnes manquantes ({len(colonnes_manquantes)}) : {apercu}...")
                else:
                    if st.button("🚀 Lancer les prédictions", type="primary"):
                        with st.spinner("Prédictions en cours..."):
                            X        = df[symptom_cols].fillna(0).astype(int)
                            X_scaled = scaler.transform(X)
                            pred     = model.predict(X_scaled)
                            maladies = label_encoder.inverse_transform(pred)
                            confiance = model.predict_proba(X_scaled).max(axis=1) * 100

                            resultats = df.copy()
                            resultats['maladie_predite'] = maladies
                            resultats['confiance_%']     = np.round(confiance, 2)

                            moy_conf = np.round(confiance.mean(), 2)

                            st.success(f"✅ {len(resultats)} prédictions réalisées ! Confiance moyenne : **{moy_conf}%**")
                            st.dataframe(resultats[['maladie_predite', 'confiance_%']].head(20), use_container_width=True)

                            csv_export = resultats.to_csv(index=False).encode('utf-8')
                            st.download_button(
                                label="📥 Télécharger les résultats CSV",
                                data=csv_export,
                                file_name="predictions_export.csv",
                                mime="text/csv",
                                type="primary"
                            )

            except Exception as e:
                st.error(f"❌ Erreur : {str(e)}")

# ── Note de bas de page ───────────────────────────────────────────────────────
st.markdown("---")
st.caption("💡 Ce système est un outil d'aide à la décision. Consultez toujours un professionnel de santé.")
