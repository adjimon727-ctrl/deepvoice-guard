# DeepVoice Guard

Prototype web de détection de médias susceptibles d’avoir été générés ou modifiés par IA. Le projet propose actuellement une interface FastAPI pour l’analyse d’images, d’audios et d’images extraites de vidéos.

> Les résultats sont des estimations de modèles, pas une preuve d’authenticité. Il faut les interpréter avec prudence et les vérifier par d’autres moyens.

## 1. Lancer le projet en local (Windows)

Installe Python 3.10 ou 3.11, puis ouvre PowerShell dans le dossier du projet :

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\python.exe -m pip install --upgrade pip
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe -m uvicorn api:app --reload --host 127.0.0.1 --port 8000
```

Ouvre ensuite <http://127.0.0.1:8000>.

Le premier démarrage télécharge les modèles depuis Hugging Face s’ils ne sont pas déjà en cache ; ces téléchargements peuvent être volumineux.

Les tests unitaires des comptes et des quotas se lancent avec `python -m unittest discover -s tests`.

## 2. État actuel

- Le modèle d’image est distinct du modèle audio ; ses classes (`Deepfake` / `Real one`) sont normalisées pour calculer les scores affichés. Les modèles sont chargés au premier usage pour alléger le démarrage. `IMAGE_MODEL_ID` et `AUDIO_MODEL_ID` permettent de choisir d’autres modèles compatibles.
- Les comptes utilisent des mots de passe hachés, des cookies de session `HttpOnly` et `SameSite=Lax`, et des endpoints d’inscription, connexion et déconnexion.
- Le quota gratuit est de 10 analyses d’images par mois. L’API requiert une connexion et applique les restrictions de formule et quotas côté serveur.
- Les offres Pro (5 000 FCFA / mois, 300 analyses) et Business (25 000 FCFA / mois, 2 000 analyses) se paient à l’unité pour un mois ; le renouvellement est manuel.
- Les paiements Paystack sont activés uniquement lorsque `PAYSTACK_SECRET_KEY` et `PUBLIC_BASE_URL` sont configurés. Le compte marchand doit accepter le XOF. Le webhook vérifie la signature et l’état de transaction côté Paystack avant d’activer une offre.
- Les fichiers envoyés sont limités à 25 Mo, écrits dans un fichier temporaire aléatoire et supprimés après traitement.
- Les formats, le mode et la résolution des images/vidéos sont vérifiés avant analyse.
- Les scores sont des sorties de classification de modèles et ne constituent pas une expertise forensique ni une preuve d’authenticité.
- L’inscription ne comprend pas encore de confirmation e-mail, de récupération de mot de passe ni de gestion des membres d’équipe.
- L’historique du navigateur est local à l’appareil et n’est pas un historique de compte.

## 3. Préparer une mise en ligne

Le serveur accepte le port de la variable d’environnement `PORT` et expose `/health` pour le contrôle de santé. [render.yaml](./render.yaml) est configuré pour une démo temporaire sur le plan gratuit ; [railway.json](./railway.json) est une configuration de départ pour Railway.

Render gratuit est uniquement pour une démonstration : le service s’arrête après 15 minutes sans trafic, le système de fichiers et la base SQLite sont effacés au redémarrage ou à l’arrêt, et la mémoire du modèle peut dépasser la limite du plan. Le premier scan peut échouer si le modèle dépasse la mémoire disponible. Ne l’utilise pas pour conserver des comptes clients ou recevoir des paiements.

Pour des données durables, configure `DATABASE_PATH` sur un stockage persistant, ainsi que `APP_ENV=production`. Les paiements nécessitent une URL publique HTTPS dans `PUBLIC_BASE_URL` et une clé `PAYSTACK_SECRET_KEY` gardée secrète. Ne configure pas les paiements sur la démo gratuite.

## 4. Étapes produit à réaliser

1. **Stabiliser et déployer** : mesurer les performances des modèles sur des exemples représentatifs, vérifier les limites de ressources, puis choisir un hébergement et un stockage persistant.
2. **Renforcer les comptes** : ajouter confirmation e-mail, récupération de mot de passe, limitation des tentatives et gestion des membres d’équipe.
3. **Valider le paiement** : vérifier l’éligibilité du compte Paystack au Bénin/XOF, configurer le webhook `/api/billing/webhook` dans le tableau de bord Paystack, puis tester une transaction en mode test avant toute activation réelle.
4. **Acquérir des clients** : contenus de démonstration en français, prospection ciblée au Bénin et en Afrique de l’Ouest, et retours d’utilisateurs pilotes.

### Exemple de lancement commercial sur 4 semaines

- **Semaine 1** : publier deux courtes démonstrations en français (une image, une arnaque vocale simulée) avec une mention claire des limites du détecteur.
- **Semaine 2** : publier une vidéo pédagogique « comment vérifier un média suspect », sans présenter les scores de l’outil comme une preuve.
- **Semaine 3** : contacter individuellement cinq médias, cabinets ou PME pour proposer un pilote gratuit et recueillir leurs retours.
- **Semaine 4** : partager les résultats du pilote avec leur accord, corriger les problèmes prioritaires et proposer une démonstration aux prospects intéressés.

Mesure les demandes de démonstration et les essais réellement terminés plutôt que les vues seules. Ne publie pas de médias privés ou d’exemples attribués à une personne sans autorisation.

Les paiements et l’activation des offres payantes exigent un compte marchand compatible, puis des secrets configurés dans l’hébergeur. Aucun secret ne doit être partagé dans le chat ou commité dans le dépôt.

## Contact professionnel

- Email : adjimon727@gmail.com
- WhatsApp : [Discuter sur WhatsApp](https://wa.me/2290167765313)
- Instagram : [@Boriso727](https://www.instagram.com/Boriso727)
