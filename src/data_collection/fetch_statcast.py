# src/data_collection/fetch_statcast.py

import pandas as pd
from pybaseball import statcast_batter
import os
from pathlib import Path
import time

def fetch_player_statcast(player_id, start_date, end_date, output_dir='data/raw/statcast'):
    """
    Fetch Statcast data for a single player and save as Parquet.
    
    Parameters:
    -----------
    player_id : int
        MLBAM player ID
    start_date : str
        Start date (YYYY-MM-DD)
    end_date : str
        End date (YYYY-MM-DD)
    output_dir : str
        Directory to save Parquet files
    
    Returns:
    --------
    dict with status info
    """
    
    # Create output directory if it doesn't exist
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    output_file = os.path.join(output_dir, f'player_{player_id}.parquet')
    
    # Check if already downloaded
    if os.path.exists(output_file):
        print(f"  Player {player_id} already downloaded, skipping...")
        return {
            'player_id': player_id,
            'success': True,
            'already_exists': True,
            'file_path': output_file
        }
    
    try:
        print(f"  Fetching Statcast data for player {player_id}...")
        
        # Fetch data
        data = statcast_batter(start_date, end_date, player_id=player_id)
        
        if data is None or len(data) == 0:
            return {
                'player_id': player_id,
                'success': False,
                'error': 'No data returned'
            }
        
        # Filter to regular season only
        if 'game_type' in data.columns:
            data = data[data['game_type'] == 'R'].copy()
        
        if len(data) == 0:
            return {
                'player_id': player_id,
                'success': False,
                'error': 'No regular season data'
            }
        
        # Save as Parquet (with compression)
        data.to_parquet(output_file, compression='snappy', index=False)
        
        file_size_mb = os.path.getsize(output_file) / (1024 * 1024)
        
        print(f"  ✓ Saved {len(data)} pitches ({file_size_mb:.2f} MB)")
        
        return {
            'player_id': player_id,
            'success': True,
            'num_pitches': len(data),
            'file_size_mb': file_size_mb,
            'file_path': output_file
        }
        
    except Exception as e:
        print(f"  ✗ Error: {e}")
        return {
            'player_id': player_id,
            'success': False,
            'error': str(e)
        }


def fetch_all_players_from_timeline(timeline_csv='data/processed/test_dates.csv', 
                                    output_dir='data/raw/statcast',
                                    delay_seconds=1):
    """
    Fetch Statcast data for all players in the timeline CSV.
    
    Parameters:
    -----------
    timeline_csv : str
        Path to CSV with columns: player_id, debut_date, date_reached
    output_dir : str
        Directory to save Parquet files
    delay_seconds : float
        Delay between API calls to avoid rate limiting
    
    Returns:
    --------
    pd.DataFrame with fetch results
    """
    
    # Load timeline
    timeline = pd.read_csv(timeline_csv)
    
    # Filter to successful players only
    timeline = timeline[timeline['success'] == True].copy()
    
    print(f"Fetching Statcast data for {len(timeline)} players...")
    print(f"Output directory: {output_dir}")
    
    results = []
    
    for idx, row in timeline.iterrows():
        player_id = int(row['player_id'])
        debut_date = row['debut_date']
        date_reached = row['date_reached']
        
        print(f"\n[{idx+1}/{len(timeline)}] Player {player_id}")
        
        # Fetch data from debut to when they reached N PA
        result = fetch_player_statcast(
            player_id=player_id,
            start_date=debut_date,
            end_date=date_reached,
            output_dir=output_dir
        )
        
        # Add player info
        result['player_name'] = row.get('player_name', 'Unknown')
        result['debut_date'] = debut_date
        result['date_reached'] = date_reached
        
        results.append(result)
        
        # Rate limiting
        if not result.get('already_exists', False):
            time.sleep(delay_seconds)
    
    # Convert to DataFrame
    results_df = pd.DataFrame(results)
    
    # Summary
    successful = results_df['success'].sum()
    total = len(results_df)
    
    print(f"\n{'='*60}")
    print(f"Fetch complete: {successful}/{total} successful")
    
    if successful > 0:
        total_size = results_df['file_size_mb'].sum()
        total_pitches = results_df['num_pitches'].sum()
        print(f"Total data: {total_pitches:,} pitches, {total_size:.2f} MB")
    
    # Save fetch log
    log_path = 'data/processed/statcast_fetch_log.csv'
    results_df.to_csv(log_path, index=False)
    print(f"\nFetch log saved to: {log_path}")
    
    return results_df


def load_player_statcast(player_id, data_dir='data/raw/statcast'):
    """
    Load a player's Statcast data from Parquet.
    
    Parameters:
    -----------
    player_id : int
        MLBAM player ID
    data_dir : str
        Directory where Parquet files are stored
    
    Returns:
    --------
    pd.DataFrame or None
    """
    file_path = os.path.join(data_dir, f'player_{player_id}.parquet')
    
    if not os.path.exists(file_path):
        print(f"File not found: {file_path}")
        return None
    
    return pd.read_parquet(file_path)


def get_storage_summary(data_dir='data/raw/statcast'):
    """
    Get summary of stored Statcast data.
    """
    
    if not os.path.exists(data_dir):
        print(f"Directory not found: {data_dir}")
        return
    
    files = list(Path(data_dir).glob('player_*.parquet'))
    
    if not files:
        print("No Parquet files found")
        return
    
    total_size = sum(f.stat().st_size for f in files)
    total_size_mb = total_size / (1024 * 1024)
    
    print(f"\n{'='*60}")
    print(f"Statcast Data Storage Summary")
    print(f"{'='*60}")
    print(f"Number of players: {len(files)}")
    print(f"Total size: {total_size_mb:.2f} MB ({total_size_mb/1024:.2f} GB)")
    print(f"Average per player: {total_size_mb/len(files):.2f} MB")
    print(f"Directory: {data_dir}")


if __name__ == "__main__":
    # Fetch all players
    results = fetch_all_players_from_timeline(
        timeline_csv='data/processed/test_dates.csv',
        output_dir='data/raw/statcast',
        delay_seconds=1
    )
    
    # Show summary
    get_storage_summary()
    
    # Test loading one player
    print("\n" + "="*60)
    print("Testing data load...")
    sample_player = results[results['success'] == True].iloc[0]['player_id']
    sample_data = load_player_statcast(sample_player)
    print(f"Loaded {len(sample_data)} pitches for player {sample_player}")
    print(sample_data.head())