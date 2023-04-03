# Import packages
import pandas as pd
import numpy as np
import requests
from bs4 import BeautifulSoup
from time import sleep
from IPython.display import clear_output
from datetime import timedelta


class SCRAPER :


    def __init__(self, years) :
        # Initiate SCRAPER object and set the scraped years as an attribute 
        self.years = years


    def scrape(self, windows = [1, 2, 5, 10, 25], save = False) :
        
        self.X = {}
        for year in self.years :

            #   1. SCRAPE THE DATA "SKELETON"
            print(f'Creating {year-1}/{year} season data skeleton ...')

            # Set URL of main webpage in BBRef of the looped season -- create a dictionary of active teams and their name code in a dictionary
            url = f'https://www.basketball-reference.com/leagues/NBA_{year}.html'
            # Create a soup object from the url
            soup = BeautifulSoup(requests.get(url).content, 'lxml')
            table = soup.find('table', id = 'per_game-team')
            df = pd.read_html(str(table))[0] # Get data of all the team which competed in the looped season
            df['Team'] = df['Team'].str.replace('*', '', regex = False) # Clean-up team names
            df = df[df['Rk'].notna()] # Get rid of average row
            # Create a list of the team codes from the hidden urls
            team_codes = pd.Series(table.find_all('a', href = True)).apply(lambda x: x['href'].split('/')[2])
            team_dic = dict(zip(df['Team'], team_codes)) # Map each team to its team code name
            sleep(1)



            #   2. SCRAPE SCHEDULE OF EACH TEAM

            data_team_home = [] # Initiate lists for all home and away games
            data_team_away = []
            n_teams = len(team_dic)
            for step, team in enumerate(list(team_dic.values())) : # Loop for every team of the season

                # Specify looped team-season url
                url = f'https://www.basketball-reference.com/teams/{team}/{year}_games.html'
                page = requests.get(url)
                soup = BeautifulSoup(page.content, 'lxml') # Create soup instance of this page
                while soup.find('tr', class_ = 'thead') is not None: # Decompose all headers
                    soup.find('tr', class_ = 'thead').decompose()

                clear_output(wait = True)
                print(f'{year-1}/{year} ({step+1}/{n_teams}) ...   Scraping {team} results ...     [{page.status_code}]')

                # Create DataFrame from the team's schedule
                df = pd.read_html(str(soup.find('table')))[0]
                # Search for the columns which contain the W/L and the venue
                for col in df.columns:
                    if '@' in set(df[col]) :
                        venue_col = col
                    if ('W' in set(df[col]) or 'L' in set(df[col])) and (col != 'Streak'):
                        win_col = col
                df = df[df['Notes'] != 'Play-In Game'] # Get rid of play-in games
                df = df.drop(columns = ['W', 'L']) # Tidy-up the data
                df = df.rename(columns = {'Tm': 'PTS', 'Opp': 'PTS_opp', win_col : 'W', venue_col: 'Venue'}) # Rename columns
                df['Team'] = len(df) * [team] # Keep track of team
                df['W'] = df['W'].replace('W', 1).replace('L', 0) # Replace Ws & Ls by 1s & 0s
                df['Date'] = pd.to_datetime(df['Date']) # Ensure dates are in datetime format
                df['Opponent'] = df['Opponent'].apply(lambda x: team_dic.get(x)) # Get opponent team code
                df['Venue'] = df['Venue'].replace(np.nan, 1).replace('@', 0) # Keep track of who played at home
                df['Streak'] = df['Streak'].apply(lambda x: {'W':1,'L':-1}.get(x.split()[0]) * float(x.split()[1])) # Transform the streak into numeric format
                df['Streak'] = [np.nan] + list(df['Streak'])[:-1] # Shift streak by one row -- for forecasting
                df = df[['Date', 'G', 'Venue', 'Team', 'Opponent', 'W', 'PTS', 'PTS_opp', 'Streak']] # Only keep necessary columns 

                # Loop through features
                for stat, underlying_stat in zip(['W/L%', 'ORtg', 'DRtg'], ['W', 'PTS', 'PTS_opp']):
                    # Calculate rolling means of features at each given window
                    df[stat] = [np.nan] + list(df[underlying_stat].rolling(1000, min_periods = 1).mean())[:-1]
                    for w in windows:
                        df[stat + '_' + str(w)] = [np.nan] + list(df[underlying_stat].rolling(w, min_periods = 1).mean())[:-1]
                # Calculate NRtg features
                df['NRtg'] = df['ORtg'] - df['DRtg']
                for w in windows :
                    df[f'NRtg_{w}'] = df[f'ORtg_{w}'] - df[f'DRtg_{w}']
                df['Rest'] = df['Date'].diff().apply(lambda x: x.total_seconds() / (24 * 3600))
                # Calculate the number of games played in the last week prior to the game
                df['Games_past_week'] = df['Date'].apply(lambda x: len(df[(df['Date'] < x) & (df['Date'] > x - timedelta(days = 7.5))])).astype(float)
                # Calculate the H2H record previous to each game
                dfs_h2h = [] # Loop for each potential opponent
                for opp in team_dic.values() :
                    if opp != team :
                        df_h2h = df.copy().groupby('Opponent').get_group(opp) # Get games against looped opponent
                        df_h2h['H2H'] = [np.nan] + list(df_h2h['W'].rolling(1000, min_periods = 1).mean())[:-1] # Calculate the rolling H2H record
                        dfs_h2h.append(df_h2h) # append "mini"-DataFrame with rolling H2H record for each opponent
                # Re-assemble the data by concatenating the "mini"-DataFrames
                df = pd.concat(dfs_h2h).sort_values('Date').reset_index(drop = True)

                # Build DataFrame of team's home games
                df_home = df.copy().groupby('Venue').get_group(1)
                df_home = df_home.drop(columns = ['Venue'])
                for col in df_home.columns: # Add a home suffix to all columns
                        df_home = df_home.rename(columns = {col: col + '_home'})
                # But, revert change for merge columns and rename some IDs columns
                df_home = df_home.rename(columns = {'Date_home': 'Date', 'Opponent_home': 'Away', 'PTS_opp_home': 'PTS_away', 'Team_home': 'Home'})
                df_home['HomeW/L%_home'] = [np.nan] + list(df_home['W_home'].rolling(1000, min_periods = 1).mean())[:-1]
                df_home['ID'] = df_home['Date'].apply(lambda x: str(x)[2:10].replace('-', '')) + df_home['Home'] + df_home['Away'] # Re-create the unique ID for each game

                # Build DataFrame of team's away games
                df_away = df.copy().groupby('Venue').get_group(0)
                df_away = df_away.drop(columns = ['Venue'])
                for col in df_away.columns: # Add an away suffix to all columns
                    df_away = df_away.rename(columns = {col: col + '_away'})
                # But, revert change for merge columns and rename some IDs columns
                df_away = df_away.rename(columns = {'Date_away': 'Date', 'Opponent_away': 'Home', 'PTS_opp_away': 'PTS_home', 'Team_away': 'Away'})
                df_away['AwayW/L%_away'] = [np.nan] + list(df_away['W_away'].rolling(1000, min_periods = 1).mean())[:-1]
                df_away['ID'] = df_away['Date'].apply(lambda x: str(x)[2:10].replace('-', '')) + df_away['Home'] + df_away['Away'] # Re-create the unique ID for each game
                # Append the team home & away DataFrame to the lists
                data_team_home.append(df_home)
                data_team_away.append(df_away)

                sleep(10) # Let's avoid getting rate limited by BBRef ...
            clear_output()


            #   3. BRING DATA ALL TEAMS TOGETHER ###
            print(f'Assembling {year-1}/{year} season final data ...')

            # Get all home stats and away stats in single DataFrame
            homes = pd.concat(data_team_home)
            aways = pd.concat(data_team_away)
                
            # Merge away and home games on the unique game ID (and overlapping columns)
            data = homes.merge(aways, on = ['ID', 'Date', 'Home', 'Away', 'PTS_home', 'PTS_away'])
            data['PTS_diff'] = data['PTS_home'] - data['PTS_away'] # Calculate the points difference for each game
            data = data.sort_values('Date').reset_index(drop = True) # Make sure data is in temporal order
            # Make a list of features
            self.features = [x for x in list(data) if 'W/L%' in x or 'ORtg' in x or 'DRtg' in x or 'Streak' in x or 'NRtg' in x or 'Rest' in x or 'Games_past_week' in x]
            self.features = self.features + ['H2H_home']
            # Re-order columns in a more readable order
            data = data[['ID', 'Date', 'G_home', 'G_away', 'Home', 'Away', 'W_home', 'PTS_home', 'PTS_away', 'PTS_diff'] + self.features]
            # Save all years data in a dictionary
            self.X[year] = data
            # If save input is True, write the DataFrame into a .csv file
            if save :
                data.to_csv(f'training_data/data_{year}.csv', index = None)