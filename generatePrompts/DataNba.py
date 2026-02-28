import os
import re

import pandas as pd

class DataNba:
    def __init__(self):
        self.matchList = []
        self.promptList = []
        self.team_names = {}
        data = loadData()
        
        if data is None:
            print("❌ Dataset NBA non chargé")
            return
        
        self.player_advanced = data.get('player_advanced')
        self.team_four_factors = data.get('team_four_factors')
        self.team_misc = data.get('team_misc')
        self.games = data.get('games')
        self.team_scoring = data.get('team_scoring')
        self.loadMatches()
        self.createPromptList()
        
    def loadMatches(self):
        if self.games is None:
            print("❌ Les données NBA (Games.csv) ne sont pas disponibles.")
            print("   Vérifiez que le fichier 'archive/Games.csv' existe.")

        # Obtenir les IDs d'équipes disponibles (hometeamId et awayteamId)
        ids_home = self.games.get('hometeamId') if 'hometeamId' in self.games.columns else pd.Series([])
        ids_away = self.games.get('awayteamId') if 'awayteamId' in self.games.columns else pd.Series([])
        # Concaténer, retirer NaN, et obtenir valeurs uniques
        team_ids = pd.concat([ids_home, ids_away]).dropna().unique().tolist()

        if len(team_ids) < 2:
            print("❌ Aucune équipe trouvée dans les données (team IDs manquants).")
            print(f"   Colonnes disponibles: {self.games.columns.tolist()}")
    

        # Limiter à 30 équipes si disponibles
        team_ids = team_ids[:30]
        
        # Créer un mapping team_id -> team_name depuis les données
        self.team_names = {}
        for team_id in team_ids:
            team_games = self.games[(self.games.get('hometeamId') == team_id) | (self.games.get('awayteamId') == team_id)]
            if not team_games.empty:
                for _, row in team_games.iterrows():
                    if row.get('hometeamId') == team_id:
                        self.team_names[team_id] = row.get('hometeamName', f'Team {team_id}')
                        break
                    elif row.get('awayteamId') == team_id:
                        self.team_names[team_id] = row.get('awayteamName', f'Team {team_id}')
                        break

        # Générer tous les matchs (chaque équipe vs chaque autre, domicile et extérieur)
        for i, home_team_id in enumerate(team_ids):
            for away_team_id in team_ids:
                if home_team_id != away_team_id:
                    self.matchList.append({
                        'home_team_id': home_team_id,
                        'away_team_id': away_team_id,
                        'home_team_name': self.team_names.get(home_team_id, f'Team {home_team_id}'),
                        'away_team_name': self.team_names.get(away_team_id, f'Team {away_team_id}')
                    })
        
        print(f"📊 Total de matchs à prédire: {len(self.matchList)} (round robin avec 30 équipes)\n")
        
    def createPromptList(self):
        for match_number, match in enumerate(self.matchList[448:], 1):
            home_team_id = match['home_team_id']
            away_team_id = match['away_team_id']
            
            # Récupérer les statistiques des deux équipes
            team1_stats = self.get_team_stats(home_team_id)
            team2_stats = self.get_team_stats(away_team_id)

            if team1_stats is None or team2_stats is None:
                print(f"⚠️ Match {match_number}: Impossible de récupérer les statistiques.")
                continue
            
            # Informations du match (le lieu est le domicile de l'équipe 1)
            match_info = {
                'lieu': f'{self.team_names.get(home_team_id, "Home")} Arena',
                'date': '2025-02-09',
                'saison': '2024-2025'
            }

            self.promptList.append(self.createPrompt(team1_stats, team2_stats, match_info))

    def get_team_stats(self, team_id):
        """Récupère les statistiques d'une équipe (par ID) pour une saison

        Cherche les matchs où `hometeamId` ou `awayteamId` == `team_id`,
        calcule victoires/défaites/points moyens et retourne aussi le nom affichable.
        """
        if self.games is None:
            return None

        # Filtrer les matchs de l'équipe (par team id)
        team_games = self.games[
            (self.games.get('hometeamId') == team_id) |
            (self.games.get('awayteamId') == team_id)
        ]

        if len(team_games) == 0:
            return None
        
        print(f"   {len(team_games)} matchs trouvés pour l'équipe ID {team_id}")

        # Trier par date et ne conserver que les 10 matchs les plus récents
        team_games = team_games.copy()
        team_games['gameDate'] = pd.to_datetime(team_games.get('gameDateTimeEst'), errors='coerce')
        team_games = team_games.sort_values('gameDate', ascending=False).head(10)
        print(f"   Utilisation des {len(team_games)} derniers matchs (les plus récents)")

        # Récupérer le nom de l'équipe (affichage) depuis le premier match trouvé
        team_name = None
        for _, row in team_games.iterrows():
            if row.get('hometeamId') == team_id:
                team_name = row.get('hometeamName', f'Team {team_id}')
                break
            if row.get('awayteamId') == team_id:
                team_name = row.get('awayteamName', f'Team {team_id}')
                break
        if team_name is None:
            team_name = f'Team {team_id}'

        # Compter victoires et accumuler points
        victoires = 0
        victoires_home = 0
        victoires_away = 0
        points_total = 0
        nb_matches = 0
        nb_home = 0
        nb_away = 0

        for _, row in team_games.iterrows():
            is_home = row.get('hometeamId') == team_id
            is_away = row.get('awayteamId') == team_id

            if is_home:
                points_total += row.get('homeScore', 0) or 0
                nb_home += 1
                winner = row.get('winner')
                if winner is not None and str(winner) == str(team_id):
                    victoires += 1
                    victoires_home += 1
            elif is_away:
                points_total += row.get('awayScore', 0) or 0
                nb_away += 1
                winner = row.get('winner')
                if winner is not None and str(winner) == str(team_id):
                    victoires += 1
                    victoires_away += 1

            nb_matches += 1

        # Calculer pourcentages de victoires
        win_home_pct = (victoires_home / nb_home) if nb_home > 0 else 0
        win_away_pct = (victoires_away / nb_away) if nb_away > 0 else 0

        # Préparer les derniers matchs en format sérialisable
        recent_games = team_games.tail(5).copy()
        if 'gameDate' in recent_games.columns:
            recent_games['gameDate'] = recent_games['gameDate'].astype(str)
        
        # Exclure les colonnes who contiennent des NaN ou ne sont pas utiles
        cols_to_drop = ['gameSubtype', 'gameLabel', 'gameSubLabel']
        recent_games = recent_games.drop(columns=[c for c in cols_to_drop if c in recent_games.columns])
        
        recent_games_records = recent_games.to_dict('records')
        advanced_avg = {}
        if self.player_advanced is not None and not self.player_advanced.empty:
            try:
                # Identifier les gameIds sélectionnés
                game_ids = team_games.get('gameId').tolist()

                # Filtrer les lignes correspondantes (prise en charge de 'teamId' ou 'team_id')
                team_col = 'teamId' if 'teamId' in self.player_advanced.columns else ('team_id' if 'team_id' in self.player_advanced.columns else None)
                if team_col is not None:
                    matched = self.player_advanced[self.player_advanced.get('gameId').isin(game_ids) & (self.player_advanced.get(team_col) == team_id)]
                else:
                    # Si pas de colonne team id, essayer de filtrer par team name
                    matched = self.player_advanced[self.player_advanced.get('gameId').isin(game_ids) & (self.player_advanced.get('teamName') == team_name)]

                if not matched.empty:
                    # Ne garder que les colonnes numériques
                    numeric_cols = matched.select_dtypes(include=['number']).columns.tolist()
                    # Exclure les colonnes identifiants, de rang, sp_work et teamName qui n'ont pas de sens en moyenne
                    numeric_cols = [c for c in numeric_cols if not re.search(r'id$|rank|person|game|teamcount|availableflag|sp_work_|teamname', c, re.I)]
                    if numeric_cols:
                        # Moyenne par match (pour éviter de pondérer par nombre de joueurs), puis moyenne sur les matchs
                        if 'gameId' in matched.columns:
                            per_game = matched.groupby('gameId')[numeric_cols].mean()
                            advanced_avg = per_game.mean().to_dict()
                        else:
                            advanced_avg = matched[numeric_cols].mean().to_dict()
            except Exception:
                advanced_avg = {}

        # Récupérer les stats des 10 derniers matchs depuis les fichiers team stats
        team_stats_avg = {}
        game_ids = team_games.get('gameId').tolist()
        
        # Stats Four Factors
        if self.team_four_factors is not None and not self.team_four_factors.empty:
            try:
                team_col = 'teamId' if 'teamId' in self.team_four_factors.columns else ('team_id' if 'team_id' in self.team_four_factors.columns else None)
                if team_col is not None:
                    matched = self.team_four_factors[self.team_four_factors.get('gameId').isin(game_ids) & (self.team_four_factors.get(team_col) == team_id)]
                    if not matched.empty:
                        numeric_cols = matched.select_dtypes(include=['number']).columns.tolist()
                        numeric_cols = [c for c in numeric_cols if not re.search(r'id$|rank|game|teamcount|availableflag', c, re.I)]
                        if numeric_cols:
                            team_stats_avg['four_factors'] = matched[numeric_cols].mean().to_dict()
            except Exception:
                pass
        
        # Stats Misc
        if self.team_misc is not None and not self.team_misc.empty:
            try:
                team_col = 'teamId' if 'teamId' in self.team_misc.columns else ('team_id' if 'team_id' in self.team_misc.columns else None)
                if team_col is not None:
                    matched = self.team_misc[self.team_misc.get('gameId').isin(game_ids) & (self.team_misc.get(team_col) == team_id)]
                    if not matched.empty:
                        numeric_cols = matched.select_dtypes(include=['number']).columns.tolist()
                        numeric_cols = [c for c in numeric_cols if not re.search(r'id$|rank|game|teamcount|availableflag', c, re.I)]
                        if numeric_cols:
                            team_stats_avg['misc'] = matched[numeric_cols].mean().to_dict()
            except Exception:
                pass
        
        # Stats Scoring
        if self.team_scoring is not None and not self.team_scoring.empty:
            try:
                team_col = 'teamId' if 'teamId' in self.team_scoring.columns else ('team_id' if 'team_id' in self.team_scoring.columns else None)
                if team_col is not None:
                    matched = self.team_scoring[self.team_scoring.get('gameId').isin(game_ids) & (self.team_scoring.get(team_col) == team_id)]
                    if not matched.empty:
                        numeric_cols = matched.select_dtypes(include=['number']).columns.tolist()
                        numeric_cols = [c for c in numeric_cols if not re.search(r'id$|rank|game|teamcount|availableflag', c, re.I)]
                        if numeric_cols:
                            team_stats_avg['scoring'] = matched[numeric_cols].mean().to_dict()
            except Exception:
                pass

        stats = {
            'team': team_name,
            'team_id': team_id,
            'matches_joues': nb_matches,
            'victoires': victoires,
            'defaites': nb_matches - victoires,
            'victoires_domicile': victoires_home,
            'defaites_domicile': nb_home - victoires_home,
            'victoires_exterieur': victoires_away,
            'defaites_exterieur': nb_away - victoires_away,
            'pourcentage_victoires_domicile': round(win_home_pct, 4),
            'pourcentage_victoires_exterieur': round(win_away_pct, 4),
            'points_moyens': (points_total / nb_matches) if nb_matches > 0 else 0,
            'derniers_matchs': recent_games_records,
            'advanced_stats_avg': advanced_avg,
            'team_stats': team_stats_avg
        }
        return stats

    def format_advanced_stats(self, stats_dict, max_items=8):
        """
        Transforme un dict de stats avancées en texte lisible pour le prompt
        """
        if not stats_dict:
            return "Non disponibles"
        items = list(stats_dict.items())[:max_items]
        return "\n".join([f"- {k}: {round(v, 3)}" for k, v in items])

    def createPrompt(self, team1_stats, team2_stats, match_info):

        team1_adv = self.format_advanced_stats(
            team1_stats.get("advanced_stats_avg", {})
        )

        team2_adv = self.format_advanced_stats(
            team2_stats.get("advanced_stats_avg", {})
        )

        team1_four = self.format_advanced_stats(
            team1_stats.get("team_stats", {}).get("four_factors", {})
        )

        team2_four = self.format_advanced_stats(
            team2_stats.get("team_stats", {}).get("four_factors", {})
        )

        return f"""
            SYSTEM:
            Tu es un modèle d'IA qui répond UNIQUEMENT en JSON.
            Ne donne aucune explication.
            Ne fais aucune phrase.
            Ne mets pas de texte avant ou après le JSON.

            Format strict :

            {{
            "equipe_gagnante": "string",
            "score_predit": "string",
            "confiance": number
            }}

            USER:

            Analyse ce match NBA.

            ÉQUIPE 1: {team1_stats.get('team')}
            - Victoires: {team1_stats.get('victoires')}
            - Défaites: {team1_stats.get('defaites')}
            - Points moyens: {team1_stats.get('points_moyens'):.1f}

            Stats avancées:
            {team1_adv}

            Four Factors:
            {team1_four}

            ---

            ÉQUIPE 2: {team2_stats.get('team')}
            - Victoires: {team2_stats.get('victoires')}
            - Défaites: {team2_stats.get('defaites')}
            - Points moyens: {team2_stats.get('points_moyens'):.1f}

            Stats avancées:
            {team2_adv}

            Four Factors:
            {team2_four}

            Contexte:
            Lieu: {match_info.get('lieu')}
            Date: {match_info.get('date')}
            Saison: {match_info.get('saison')}

            Réponds uniquement avec le JSON.
        """

    def getPromptList(self):
        return self.promptList

def loadData():
    """Charge tous les fichiers CSV du dataset NBA"""
    try:
        print("📊 Chargement des données NBA depuis le dossier archive...")
        
        data = {}
        files = {
            'players': 'Players.csv',
            'team_statistics': 'TeamStatistics.csv',
            'player_statistics': 'PlayerStatistics.csv',
            'games': 'Games.csv',
            'team_histories': 'TeamHistories.csv',
            'league_schedule': 'LeagueSchedule24_25.csv',
            'player_advanced': 'PlayerStatisticsAdvanced.csv',
            'team_four_factors': 'TeamStatisticsFourFactors.csv',
            'team_misc': 'TeamStatisticsMisc.csv',
            'team_scoring': 'TeamStatisticsScoring.csv'
        }
        
        for key, filename in files.items():
            filepath = os.path.join("archive", filename)
            if os.path.exists(filepath):
                try:
                    data[key] = pd.read_csv(filepath)
                    print(f"  ✅ {filename}: {len(data[key])} lignes")
                except Exception as e:
                    print(f"  ⚠️ Erreur chargement {filename}: {e}")
                    data[key] = None
            else:
                print(f"  ⚠️ {filename} non trouvé")
                data[key] = None
        
        print(f"\n✅ Tous les fichiers chargés")
        return data
            
    except Exception as e:
        print(f"⚠️ Erreur lors du chargement: {e}")
        return None
    
def loadPrompt():
    """Fonction principale"""
    print("🏀 Générateur de prédictions NBA avec IA (Round Robin: 30 équipes)\n")
    
    # Charger TOUS les fichiers CSV au début
    data = loadData()
    
    if data is None:
        print("❌ Les données NBA ne sont pas disponibles.")
        return
    
    return data
    
    