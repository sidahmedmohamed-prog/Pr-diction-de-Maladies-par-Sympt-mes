import pandas as pd
import numpy as np
import joblib
import gradio as gr
import os

# Chargement des modèles
model = joblib.load('model_rf_tuned.pkl')
scaler = joblib.load('scaler.pkl')
label_encoder = joblib.load('label_encoder.pkl')
symptom_cols = joblib.load('symptom_cols.pkl')

# ─────────────────────────────────────────────
# Fonction 1 : Prédiction individuelle
# ─────────────────────────────────────────────
def predire_individuel(symptomes):
    if symptomes is None or len(symptomes) == 0:
        return '⚠️ Veuillez sélectionner au moins un symptôme.'

    vecteur = np.zeros(len(symptom_cols))
    for s in symptomes:
        if s in symptom_cols:
            vecteur[symptom_cols.index(s)] = 1

    X = scaler.transform([vecteur])
    pred = model.predict(X)[0]
    maladie = label_encoder.inverse_transform([pred])[0]

    proba = model.predict_proba(X)[0]
    top3_idx = np.argsort(proba)[::-1][:3]

    texte = f'✅ Maladie prédite : {maladie}\n\n'
    texte += '📊 Top 3 probabilités :\n'
    for i, idx in enumerate(top3_idx, 1):
        nom = label_encoder.classes_[idx]
        pct = proba[idx] * 100
        texte += f'  {i}. {nom} : {pct:.2f}%\n'

    texte += f'\n🔢 Symptômes sélectionnés : {len(symptomes)}'
    return texte

# ─────────────────────────────────────────────
# Fonction 2 : Prédiction par fichier
# ─────────────────────────────────────────────
def predire_par_fichier(fichier):
    if fichier is None:
        return None, '⚠️ Veuillez importer un fichier CSV ou Excel.'

    path = fichier.name
    try:
        if path.endswith('.csv'):
            df = pd.read_csv(path)
        elif path.endswith(('.xlsx', '.xls')):
            df = pd.read_excel(path)
        else:
            return None, '❌ Format non supporté. Utilisez CSV ou Excel (.csv, .xlsx, .xls).'
    except Exception as e:
        return None, f'❌ Erreur lors de la lecture du fichier : {str(e)}'

    colonnes_manquantes = [c for c in symptom_cols if c not in df.columns]
    if colonnes_manquantes:
        apercu = ', '.join(colonnes_manquantes[:5])
        return None, f'❌ Colonnes manquantes ({len(colonnes_manquantes)}) : {apercu}...'

    try:
        X = df[symptom_cols].fillna(0).astype(int)
        X_scaled = scaler.transform(X)
        pred = model.predict(X_scaled)
        maladies = label_encoder.inverse_transform(pred)
        confiance = model.predict_proba(X_scaled).max(axis=1) * 100

        resultats = df.copy()
        resultats['maladie_predite'] = maladies
        resultats['confiance_%'] = np.round(confiance, 2)

        output_path = 'predictions_export.csv'
        resultats.to_csv(output_path, index=False)

        moy_conf = np.round(confiance.mean(), 2)
        return output_path, f'✅ {len(resultats)} prédictions réalisées avec succès.\n📈 Confiance moyenne : {moy_conf}%'
    except Exception as e:
        return None, f'❌ Erreur lors de la prédiction : {str(e)}'

# ─────────────────────────────────────────────
# Interface Gradio
# ─────────────────────────────────────────────
css = """
body { font-family: 'Segoe UI', sans-serif; }
.gradio-container { max-width: 960px !important; margin: auto; }
.tab-nav button { font-size: 16px; font-weight: 600; }
#result_box textarea { font-size: 15px; line-height: 1.7; }
footer { display: none !important; }
"""

with gr.Blocks(title='🩺 Prédiction de Maladies') as demo:

    gr.Markdown("""
    # 🩺 Prédiction de Maladies par Symptômes
    **Projet Machine Learning** — Classification multi-classe de **42 maladies**  
    Modèle : Random Forest optimisé | Données : symptômes cliniques binaires
    ---
    """)

    with gr.Tab('🧑 Test individuel'):
        gr.Markdown("### Sélectionnez les symptômes du patient")
        symptomes = gr.CheckboxGroup(
            choices=symptom_cols,
            label='Symptômes présents',
            info='Cochez tous les symptômes observés chez le patient.'
        )
        with gr.Row():
            bouton_reset = gr.Button('🔄 Réinitialiser', variant='secondary')
            bouton = gr.Button('🔍 Prédire la maladie', variant='primary', scale=2)

        sortie = gr.Textbox(
            label='Résultat de la prédiction',
            lines=8,
            elem_id='result_box',
            placeholder='Le résultat apparaîtra ici après la prédiction...'
        )

        bouton.click(predire_individuel, inputs=symptomes, outputs=sortie)
        bouton_reset.click(lambda: ([], ''), outputs=[symptomes, sortie])

    with gr.Tab('📂 Test par fichier CSV / Excel'):
        gr.Markdown("""
        ### Importez un fichier pour prédire plusieurs patients
        Le fichier doit contenir **une colonne par symptôme** (valeurs 0 ou 1).
        """)
        fichier = gr.File(
            label='📎 Importer un fichier CSV ou Excel',
            file_types=['.csv', '.xlsx', '.xls']
        )
        bouton_fichier = gr.Button('🚀 Lancer les prédictions', variant='primary')

        with gr.Row():
            fichier_sortie = gr.File(label='📥 Télécharger les résultats')
            message = gr.Textbox(label='Statut', lines=4)

        bouton_fichier.click(
            predire_par_fichier,
            inputs=fichier,
            outputs=[fichier_sortie, message]
        )

    gr.Markdown("""
    ---
    > 💡 **Note** : Ce système est un outil d'aide à la décision. Consultez toujours un professionnel de santé.
    """)

if __name__ == '__main__':
    demo.launch(
        server_name="127.0.0.1",
        server_port=None,        # Gradio choisit le port libre automatiquement
        css=css,
        theme=gr.themes.Soft()
    )
