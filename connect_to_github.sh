#!/bin/bash

# Script per collegare il repository locale a GitHub
# Esegui questo script dopo aver creato la repository su GitHub

echo "=========================================="
echo "Collegamento repository a GitHub"
echo "=========================================="
echo ""
echo "Inserisci le informazioni della repository che hai appena creato:"
echo ""

read -p "Nome utente GitHub: " GITHUB_USERNAME
read -p "Nome della repository: " REPO_NAME

if [ -z "$GITHUB_USERNAME" ] || [ -z "$REPO_NAME" ]; then
    echo "Errore: Nome utente e nome repository sono obbligatori!"
    exit 1
fi

echo ""
echo "Collego il repository locale a GitHub..."
echo ""

# Aggiungi il remote
git remote add origin "https://github.com/${GITHUB_USERNAME}/${REPO_NAME}.git" 2>/dev/null

if [ $? -ne 0 ]; then
    echo "Il remote 'origin' esiste già. Vuoi sovrascriverlo? (s/n)"
    read -p "> " answer
    if [ "$answer" = "s" ] || [ "$answer" = "S" ]; then
        git remote remove origin
        git remote add origin "https://github.com/${GITHUB_USERNAME}/${REPO_NAME}.git"
    else
        echo "Operazione annullata."
        exit 1
    fi
fi

# Assicurati di essere sul branch main
git branch -M main

# Verifica la connessione
echo ""
echo "Verifico la connessione..."
git remote -v

echo ""
echo "Eseguo il push del codice su GitHub..."
echo ""

# Push del codice
git push -u origin main

if [ $? -eq 0 ]; then
    echo ""
    echo "=========================================="
    echo "✅ Successo! Il codice è stato caricato su GitHub"
    echo "=========================================="
    echo ""
    echo "Puoi visualizzare la repository qui:"
    echo "https://github.com/${GITHUB_USERNAME}/${REPO_NAME}"
    echo ""
else
    echo ""
    echo "=========================================="
    echo "❌ Errore durante il push"
    echo "=========================================="
    echo ""
    echo "Possibili cause:"
    echo "- La repository non esiste ancora su GitHub"
    echo "- Problemi di autenticazione (potresti dover inserire username/password o token)"
    echo "- La repository esiste ma non è vuota"
    echo ""
    echo "Se devi autenticarti, GitHub potrebbe richiedere:"
    echo "- Username: ${GITHUB_USERNAME}"
    echo "- Password: usa un Personal Access Token (non la password di GitHub)"
    echo ""
    echo "Per creare un token: https://github.com/settings/tokens"
fi



