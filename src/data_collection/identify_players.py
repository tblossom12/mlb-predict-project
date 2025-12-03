# src/data_collection/identify_players.py

from pybaseball import statcast_batter, batting_stats, playerid_reverse_lookup
import pandas as pd
from datetime import datetime

def find_nth_pa_date(player_id, n_pa=500, start_date='2015-01-01', end_date='2024-12-31'):
    """
    Find the date when a player reached their Nth plate appearance in REGULAR SEASON.
    
    Parameters:
    -----------
    player_id : int
        MLB player ID (MLBAM ID)
    n_pa : int
        Target number of plate appearances (default 500)
    start_date : str
        Start date for searching (YYYY-MM-DD)
    end_date : str
        End date for searching (YYYY-MM-DD)
    
    Returns:
    --------
    dict with player PA timeline info
    """
    
    try:
        print(f"Fetching data for player {player_id}...")
        
        # Get all Statcast data for the player
        data = statcast_batter(start_date, end_date, player_id=player_id)
        
        if data is None or len(data) == 0:
            return {
                'player_id': player_id,
                'n_pa': n_pa,
                'date_reached': None,
                'total_pas_found': 0,
                'debut_date': None,
                'success': False,
                'error': 'No data found'
            }
        
        # FILTER TO REGULAR SEASON ONLY
        # Statcast has a 'game_type' column: 'R' = Regular Season, 'S' = Spring Training, etc.
        if 'game_type' in data.columns:
            regular_season = data[data['game_type'].isin(['R', 'F', 'D', 'L', 'W'])]
            print(f"  Filtered from {len(data)} total pitches to {len(regular_season)} regular season pitches")
        else:
            # Fallback: filter by date (regular season typically starts late March/early April)
            print("  Warning: 'game_type' column not found, filtering by date...")
            regular_season = data.copy()
            # Remove February and early March (spring training)
            regular_season = regular_season[
                ~((regular_season['game_date'].dt.month < 3) | 
                  ((regular_season['game_date'].dt.month == 3) & (regular_season['game_date'].dt.day < 20)))
            ]
        
        if len(regular_season) == 0:
            return {
                'player_id': player_id,
                'n_pa': n_pa,
                'date_reached': None,
                'total_pas_found': 0,
                'debut_date': None,
                'success': False,
                'error': 'No regular season data found'
            }
        
        # Sort chronologically
        regular_season = regular_season.sort_values(['game_date', 'inning', 'at_bat_number'])
        
        # Identify unique plate appearances
        regular_season['pa_id'] = regular_season.groupby(['game_date', 'at_bat_number']).ngroup()
        
        # Get one row per PA (take last pitch of each PA which has the outcome)
        pas = regular_season.groupby('pa_id').last().reset_index()
        
        # Sort by date to ensure chronological order
        pas = pas.sort_values('game_date')
        
        total_pas = len(pas)
        debut_date = pas['game_date'].min()
        
        # Check if player reached n_pa
        if total_pas < n_pa:
            return {
                'player_id': player_id,
                'n_pa': n_pa,
                'date_reached': None,
                'total_pas_found': total_pas,
                'debut_date': str(debut_date),
                'success': False,
                'error': f'Only reached {total_pas} PAs in regular season'
            }
        
        # Get the nth PA
        nth_pa = pas.iloc[n_pa - 1]  # -1 because 0-indexed
        date_reached = nth_pa['game_date']
        
        return {
            'player_id': player_id,
            'n_pa': n_pa,
            'date_reached': str(date_reached),
            'total_pas_found': total_pas,
            'debut_date': str(debut_date),
            'success': True
        }
        
    except Exception as e:
        return {
            'player_id': player_id,
            'n_pa': n_pa,
            'date_reached': None,
            'total_pas_found': 0,
            'debut_date': None,
            'success': False,
            'error': str(e)
        }


def get_player_id_mapping():
    """
    Get mapping between FanGraphs IDs and MLBAM IDs using Chadwick Bureau register.
    
    Returns:
    --------
    pd.DataFrame with columns: key_fangraphs, key_mlbam, name_first, name_last
    """
    from pybaseball import chadwick_register
    
    print("Loading Chadwick Bureau player register...")
    register = chadwick_register()
    
    # Filter to players with both FanGraphs and MLBAM IDs
    mapping = register[
        (register['key_fangraphs'].notna()) & 
        (register['key_mlbam'].notna())
    ][['key_fangraphs', 'key_mlbam', 'name_first', 'name_last']].copy()
    
    # Convert IDs to integers
    mapping['key_fangraphs'] = mapping['key_fangraphs'].astype(int)
    mapping['key_mlbam'] = mapping['key_mlbam'].astype(int)
    
    print(f"Loaded {len(mapping)} player ID mappings")
    
    return mapping


def find_qualifying_players(min_career_pa=1000, debut_year_start=2015, debut_year_end=2020):
    """
    Simplified version - just look a couple years before debut range
    """
    
    print(f"Finding players who debuted {debut_year_start}-{debut_year_end} with {min_career_pa}+ PAs...")
    
    # Only go back 2-3 years before debut range (most debuts are identified quickly)
    # This reduces amount of data we need to fetch
    data_start_year = max(debut_year_start - 3, 2010)  # Don't go before 2010
    
    print(f"Fetching batting data from {data_start_year} to 2024...")
    
    try:
        all_data = batting_stats(data_start_year, 2024, qual=1)
    except Exception as e:
        print(f"Error fetching data: {e}")
        print("Trying with smaller range...")
        # Fallback: just get debut years + a few after
        all_data = batting_stats(debut_year_start, 2024, qual=1)
    
    # Group by player
    player_stats = all_data.groupby('IDfg').agg({
        'Name': 'first',
        'Season': ['min', 'max'],
        'PA': 'sum'
    }).reset_index()
    
    player_stats.columns = ['player_id_fg', 'player_name', 'debut_year', 'last_year', 'total_pa']
    
    # Filter to target range
    qualifying = player_stats[
        (player_stats['debut_year'] >= debut_year_start) &
        (player_stats['debut_year'] <= debut_year_end) &
        (player_stats['total_pa'] >= min_career_pa)
    ].copy()
    
    print(f"\nFound {len(qualifying)} players")
    
    if len(qualifying) > 0:
        print(f"Debut year range: {qualifying['debut_year'].min()}-{qualifying['debut_year'].max()}")
    
    # Get ID mapping
    id_mapping = get_player_id_mapping()
    
    qualifying = qualifying.merge(
        id_mapping[['key_fangraphs', 'key_mlbam']],
        left_on='player_id_fg',
        right_on='key_fangraphs',
        how='left'
    )
    
    qualifying = qualifying.rename(columns={'key_mlbam': 'mlbam_id'})
    qualifying = qualifying.drop(columns=['key_fangraphs'])
    
    before_filter = len(qualifying)
    qualifying = qualifying[qualifying['mlbam_id'].notna()].copy()
    after_filter = len(qualifying)
    
    print(f"Matched {after_filter}/{before_filter} players to MLBAM IDs")
    
    if len(qualifying) > 0:
        print("\nSample:")
        print(qualifying[['player_name', 'debut_year', 'total_pa']].head(10).to_string(index=False))
    
    return qualifying

def build_player_timeline(min_career_pa=1000, n_pa=500, debut_year_start=2015, debut_year_end=2020):
    """
    Complete pipeline to identify all qualifying players and find their Nth PA dates.
    
    Returns:
    --------
    pd.DataFrame with player info and key dates
    """
    
    # Step 1: Find qualifying players (already includes MLBAM ID mapping)
    qualifying_players = find_qualifying_players(min_career_pa, debut_year_start, debut_year_end)
    
    if len(qualifying_players) == 0:
        print("No qualifying players found!")
        return pd.DataFrame()
    
    # Step 2: Find Nth PA date for each player
    results = []
    
    for idx, row in qualifying_players.iterrows():
        player_id = int(row['mlbam_id'])
        player_name = row['player_name']
        
        print(f"\nProcessing {player_name} (MLBAM ID: {player_id})...")
        
        # Find when they reached n_pa
        result = find_nth_pa_date(
            player_id=player_id,
            n_pa=n_pa,
            start_date=f"{row['debut_year']}-01-01",
            end_date="2024-12-31"
        )
        
        # Add player info
        result['player_name'] = player_name
        result['fg_id'] = row['player_id_fg']
        result['debut_year'] = row['debut_year']
        result['total_career_pa'] = row['total_pa']
        
        results.append(result)
        
        # Optional: Add small delay to avoid rate limiting
        import time
        time.sleep(0.5)
    
    # Convert to DataFrame
    timeline_df = pd.DataFrame(results)
    
    # Filter to successful cases
    successful = timeline_df[timeline_df['success'] == True]
    
    print(f"\n{'='*50}")
    print(f"Successfully found {n_pa}th PA date for {len(successful)}/{len(timeline_df)} players")
    
    return timeline_df


# Test function
if __name__ == "__main__":
    # Test with one known player first
    print("Testing with Shohei Ohtani (MLBAM ID: 660271)...")
    result = find_nth_pa_date(player_id=660271, n_pa=500, start_date='2018-01-01')
    print(result)
    
    # If that works, try the full pipeline with small sample
    if result['success']:
        print("\n" + "="*50)
        print("Running pipeline for debut year 2018 only (small test)...")
        timeline = build_player_timeline(
            min_career_pa=1000,
            n_pa=500,
            debut_year_start=2018,
            debut_year_end=2018
        )
        
        if len(timeline) > 0:
            print("\nResults:")
            print(timeline[['player_name', 'debut_date', 'date_reached', 'total_pas_found', 'success']])
            
            # Save results
            timeline.to_csv('data/processed/player_timeline.csv', index=False)
            print("\nSaved to data/processed/player_timeline.csv")
        else:
            print("\nNo players found in timeline!")