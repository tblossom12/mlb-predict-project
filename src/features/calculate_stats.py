# Calculate batting stats from Statcast
# src/features/calculate_stats.py

import pandas as pd
import numpy as np

class StatcastFeatureCalculator:
    """
    Calculate batting features from Statcast pitch-level data.
    Uses mix of actual results and expected stats (xStats) based on quality of contact.
    """
    
    def __init__(self, pitch_data):
        """
        Initialize with pitch-level Statcast data for a single player.
        
        Parameters:
        -----------
        pitch_data : pd.DataFrame
            Raw Statcast pitch data for one player
        """
        self.pitch_data = pitch_data.copy()
        
        # Create plate appearance level data
        self.pitch_data = self.pitch_data.sort_values(['game_date', 'inning', 'at_bat_number'])
        self.pas = self.pitch_data.groupby(['game_date', 'at_bat_number']).last().reset_index()
        
        # Get events (PA outcomes)
        self.events = self.pas['events'].dropna()
        
        # Batted ball data
        self.batted_balls = self.pas[self.pas['launch_speed'].notna()].copy()
        
    def calculate_all_features(self):
        """
        Calculate all 28 features.
        
        Returns:
        --------
        dict : All features with their values
        """
        features = {}
        
        # Basic features (14) - mix of actual and expected
        features.update(self._basic_features())
        
        # Advanced Statcast features (14)
        features.update(self._advanced_features())
        
        return features
    
    # ==================== BASIC FEATURES (14) ====================
    
    def _basic_features(self):
        """Calculate 14 basic features (actual results + xStats)"""
        return {
            'AVG': self._avg(),
            'OBP': self._obp(),
            'SLG': self._slg(),
            'OPS': self._ops(),
            'ISO': self._iso(),
            'BB%': self._bb_rate(),
            'K%': self._k_rate(),
            'BABIP': self._babip(),
            'HR/FB': self._hr_fb(),
            'LD%': self._ld_rate(),
            'GB%': self._gb_rate(),
            'FB%': self._fb_rate(),
            'xwOBA': self._xwoba(),      # Expected wOBA
            'xwRC+': self._xwrc_plus(),  # Expected wRC+
        }
    
    def _avg(self):
        """Batting Average (actual)"""
        hits = self.events.isin(['single', 'double', 'triple', 'home_run']).sum()
        non_ab = ['walk', 'hit_by_pitch', 'sac_fly', 'sac_bunt', 'catcher_interf']
        at_bats = (~self.events.isin(non_ab)).sum()
        return hits / at_bats if at_bats > 0 else 0
    
    def _obp(self):
        """On-Base Percentage (actual)"""
        hits = self.events.isin(['single', 'double', 'triple', 'home_run']).sum()
        walks = (self.events == 'walk').sum()
        hbp = (self.events == 'hit_by_pitch').sum()
        return (hits + walks + hbp) / len(self.events) if len(self.events) > 0 else 0
    
    def _slg(self):
        """Slugging Percentage (actual)"""
        singles = (self.events == 'single').sum()
        doubles = (self.events == 'double').sum()
        triples = (self.events == 'triple').sum()
        home_runs = (self.events == 'home_run').sum()
        total_bases = singles + 2*doubles + 3*triples + 4*home_runs
        
        non_ab = ['walk', 'hit_by_pitch', 'sac_fly', 'sac_bunt', 'catcher_interf']
        at_bats = (~self.events.isin(non_ab)).sum()
        return total_bases / at_bats if at_bats > 0 else 0
    
    def _ops(self):
        """On-Base Plus Slugging (actual)"""
        return self._obp() + self._slg()
    
    def _iso(self):
        """Isolated Power (actual)"""
        return self._slg() - self._avg()
    
    def _bb_rate(self):
        """Walk Rate (BB%)"""
        walks = (self.events == 'walk').sum()
        return walks / len(self.events) if len(self.events) > 0 else 0
    
    def _k_rate(self):
        """Strikeout Rate (K%)"""
        strikeouts = self.events.isin(['strikeout', 'strikeout_double_play']).sum()
        return strikeouts / len(self.events) if len(self.events) > 0 else 0
    
    def _babip(self):
        """Batting Average on Balls In Play (actual)"""
        hits = self.events.isin(['single', 'double', 'triple']).sum()
        non_babip_events = ['walk', 'hit_by_pitch', 'sac_fly', 'sac_bunt', 'catcher_interf',
                           'strikeout', 'strikeout_double_play', 'home_run']
        balls_in_play = (~self.events.isin(non_babip_events)).sum()
        return hits / balls_in_play if balls_in_play > 0 else 0
    
    def _hr_fb(self):
        """Home Run to Fly Ball Ratio"""
        if 'bb_type' not in self.batted_balls.columns:
            return 0
        fly_balls = self.batted_balls['bb_type'].isin(['fly_ball', 'popup']).sum()
        home_runs = (self.events == 'home_run').sum()
        return home_runs / fly_balls if fly_balls > 0 else 0
    
    def _ld_rate(self):
        """Line Drive Rate (LD%)"""
        if 'bb_type' not in self.batted_balls.columns:
            return 0
        line_drives = (self.batted_balls['bb_type'] == 'line_drive').sum()
        return line_drives / len(self.batted_balls) if len(self.batted_balls) > 0 else 0
    
    def _gb_rate(self):
        """Ground Ball Rate (GB%)"""
        if 'bb_type' not in self.batted_balls.columns:
            return 0
        ground_balls = (self.batted_balls['bb_type'] == 'ground_ball').sum()
        return ground_balls / len(self.batted_balls) if len(self.batted_balls) > 0 else 0
    
    def _fb_rate(self):
        """Fly Ball Rate (FB%)"""
        if 'bb_type' not in self.batted_balls.columns:
            return 0
        fly_balls = self.batted_balls['bb_type'].isin(['fly_ball', 'popup']).sum()
        return fly_balls / len(self.batted_balls) if len(self.batted_balls) > 0 else 0
    
    def _xwoba(self):
        """
        Expected Weighted On-Base Average (xwOBA)
        Based on quality of contact (exit velo + launch angle), not actual outcomes.
        Uses Statcast's estimated_woba_using_speedangle.
        """
        if 'estimated_woba_using_speedangle' in self.pas.columns:
            xwoba_values = self.pas['estimated_woba_using_speedangle'].dropna()
            if len(xwoba_values) > 0:
                return xwoba_values.mean()
        
        # Fallback: use woba_value and woba_denom columns
        if 'woba_value' in self.pas.columns and 'woba_denom' in self.pas.columns:
            total_value = self.pas['woba_value'].sum()
            total_denom = self.pas['woba_denom'].sum()
            if total_denom > 0:
                return total_value / total_denom
        
        # Last resort: return 0 (can't calculate without Statcast data)
        return 0
    
    def _xwrc_plus(self):
        """
        Expected Weighted Runs Created Plus (xwRC+)
        Based on xwOBA, normalized to 100 (league average).
        Measures expected runs created per PA relative to league average.
        """
        xwoba = self._xwoba()
        
        if xwoba == 0:
            return 100
        
        # MLB average wOBA (approximate - ideally would match player's season/era)
        # 2023: ~0.320, 2022: ~0.317, 2021: ~0.320
        league_woba = 0.320
        
        # wOBA scale factor (converts wOBA to runs per PA)
        # Changes slightly by year but ~1.25 is typical
        woba_scale = 1.25
        
        # wRC+ formula: ((wOBA - lgwOBA) / wOBA_scale) * (1 / lgwOBA) * 100 + 100
        # This gives 100 = average, >100 = above average, <100 = below average
        xwrc_plus = ((xwoba - league_woba) / woba_scale) / league_woba * 100 + 100
        
        return max(0, xwrc_plus)  # Don't return negative
    
    # ==================== ADVANCED STATCAST FEATURES (14) ====================
    
    def _advanced_features(self):
        """Calculate 14 advanced Statcast features"""
        return {
            'Hard%': self._hard_hit_rate(),
            'EV': self._avg_exit_velo(),
            'maxEV': self._max_exit_velo(),
            'Barrel%': self._barrel_rate(),
            'LA': self._avg_launch_angle(),
            'Sweet Spot%': self._sweet_spot_rate(),
            'O-Swing%': self._o_swing_rate(),
            'Z-Swing%': self._z_swing_rate(),
            'SwStr%': self._swinging_strike_rate(),
            'O-Contact%': self._o_contact_rate(),
            'Z-Contact%': self._z_contact_rate(),
            'Pull%': self._pull_rate(),
            'Cent%': self._center_rate(),
            'Oppo%': self._oppo_rate(),
        }
    
    def _avg_exit_velo(self):
        """Average Exit Velocity (EV) in mph"""
        if len(self.batted_balls) == 0:
            return 0
        return self.batted_balls['launch_speed'].mean()
    
    def _max_exit_velo(self):
        """Maximum Exit Velocity (maxEV) in mph"""
        if len(self.batted_balls) == 0:
            return 0
        return self.batted_balls['launch_speed'].max()
    
    def _avg_launch_angle(self):
        """Average Launch Angle (LA) in degrees"""
        if len(self.batted_balls) == 0:
            return 0
        return self.batted_balls['launch_angle'].mean()
    
    def _hard_hit_rate(self):
        """Hard Hit Rate (Hard%) - batted balls ≥95 mph"""
        if len(self.batted_balls) == 0:
            return 0
        hard_hit = (self.batted_balls['launch_speed'] >= 95).sum()
        return hard_hit / len(self.batted_balls)
    
    def _barrel_rate(self):
        """
        Barrel Rate (Barrel%) - per PA
        Barrels = optimal combination of exit velo + launch angle
        """
        if len(self.pas) == 0:
            return 0
        
        # Use Statcast's launch_speed_angle column (6 = barrel)
        if 'launch_speed_angle' in self.batted_balls.columns:
            barrels = (self.batted_balls['launch_speed_angle'] == 6).sum()
            return barrels / len(self.pas)
        
        # Fallback: simplified barrel definition
        # 98+ mph EV with 26-30° launch angle
        if len(self.batted_balls) == 0:
            return 0
        
        barrels = self.batted_balls[
            (self.batted_balls['launch_speed'] >= 98) &
            (self.batted_balls['launch_angle'] >= 26) &
            (self.batted_balls['launch_angle'] <= 30)
        ]
        return len(barrels) / len(self.pas)
    
    def _sweet_spot_rate(self):
        """Sweet Spot% - batted balls with 8-32° launch angle"""
        if len(self.batted_balls) == 0:
            return 0
        sweet_spot = self.batted_balls[
            (self.batted_balls['launch_angle'] >= 8) &
            (self.batted_balls['launch_angle'] <= 32)
        ]
        return len(sweet_spot) / len(self.batted_balls)
    
    def _o_swing_rate(self):
        """Outside Zone Swing Rate (O-Swing%)"""
        if 'zone' not in self.pitch_data.columns or 'description' not in self.pitch_data.columns:
            return 0
        
        # Zones 11-14 are outside strike zone
        outside_zone = self.pitch_data['zone'].isin([11, 12, 13, 14])
        outside_pitches = self.pitch_data[outside_zone]
        
        if len(outside_pitches) == 0:
            return 0
        
        swings = outside_pitches['description'].isin([
            'foul', 'hit_into_play', 'swinging_strike', 
            'swinging_strike_blocked', 'foul_tip'
        ]).sum()
        
        return swings / len(outside_pitches)
    
    def _z_swing_rate(self):
        """Inside Zone Swing Rate (Z-Swing%)"""
        if 'zone' not in self.pitch_data.columns or 'description' not in self.pitch_data.columns:
            return 0
        
        # Zones 1-9 are inside strike zone
        inside_zone = self.pitch_data['zone'].isin(range(1, 10))
        inside_pitches = self.pitch_data[inside_zone]
        
        if len(inside_pitches) == 0:
            return 0
        
        swings = inside_pitches['description'].isin([
            'foul', 'hit_into_play', 'swinging_strike', 
            'swinging_strike_blocked', 'foul_tip'
        ]).sum()
        
        return swings / len(inside_pitches)
    
    def _swinging_strike_rate(self):
        """Swinging Strike Rate (SwStr%) - per pitch"""
        if 'description' not in self.pitch_data.columns:
            return 0
        
        whiffs = self.pitch_data['description'].isin([
            'swinging_strike', 'swinging_strike_blocked'
        ]).sum()
        
        return whiffs / len(self.pitch_data)
    
    def _o_contact_rate(self):
        """Outside Zone Contact Rate (O-Contact%)"""
        if 'zone' not in self.pitch_data.columns or 'description' not in self.pitch_data.columns:
            return 0
        
        outside_zone = self.pitch_data['zone'].isin([11, 12, 13, 14])
        outside_swings = self.pitch_data[
            outside_zone & 
            self.pitch_data['description'].isin([
                'foul', 'hit_into_play', 'swinging_strike', 
                'swinging_strike_blocked', 'foul_tip'
            ])
        ]
        
        if len(outside_swings) == 0:
            return 0
        
        contact = outside_swings['description'].isin(['foul', 'hit_into_play', 'foul_tip']).sum()
        
        return contact / len(outside_swings)
    
    def _z_contact_rate(self):
        """Inside Zone Contact Rate (Z-Contact%)"""
        if 'zone' not in self.pitch_data.columns or 'description' not in self.pitch_data.columns:
            return 0
        
        inside_zone = self.pitch_data['zone'].isin(range(1, 10))
        inside_swings = self.pitch_data[
            inside_zone & 
            self.pitch_data['description'].isin([
                'foul', 'hit_into_play', 'swinging_strike', 
                'swinging_strike_blocked', 'foul_tip'
            ])
        ]
        
        if len(inside_swings) == 0:
            return 0
        
        contact = inside_swings['description'].isin(['foul', 'hit_into_play', 'foul_tip']).sum()
        
        return contact / len(inside_swings)
    
    def _pull_rate(self):
        """Pull% - based on horizontal hit location"""
        if 'hc_x' not in self.batted_balls.columns or 'stand' not in self.batted_balls.columns:
            return 0
        
        if len(self.batted_balls) == 0:
            return 0
        
        pulled = 0
        for _, row in self.batted_balls.iterrows():
            if pd.notna(row['hc_x']) and pd.notna(row['stand']):
                # hc_x < 125 is pull side for righties, > 125 for lefties
                if (row['stand'] == 'R' and row['hc_x'] < 125) or \
                   (row['stand'] == 'L' and row['hc_x'] > 125):
                    pulled += 1
        
        return pulled / len(self.batted_balls)
    
    def _center_rate(self):
        """Center% - balls hit to center field"""
        if 'hc_x' not in self.batted_balls.columns:
            return 0
        
        if len(self.batted_balls) == 0:
            return 0
        
        # Approximately 100-150 on hc_x scale
        center = self.batted_balls[
            (self.batted_balls['hc_x'] >= 100) &
            (self.batted_balls['hc_x'] <= 150)
        ]
        
        return len(center) / len(self.batted_balls)
    
    def _oppo_rate(self):
        """Opposite Field% - balls hit opposite way"""
        if 'hc_x' not in self.batted_balls.columns or 'stand' not in self.batted_balls.columns:
            return 0
        
        if len(self.batted_balls) == 0:
            return 0
        
        oppo = 0
        for _, row in self.batted_balls.iterrows():
            if pd.notna(row['hc_x']) and pd.notna(row['stand']):
                # Opposite of pull
                if (row['stand'] == 'R' and row['hc_x'] > 125) or \
                   (row['stand'] == 'L' and row['hc_x'] < 125):
                    oppo += 1
        
        return oppo / len(self.batted_balls)


# ==================== CONVENIENCE FUNCTIONS ====================

def calculate_player_features(player_id, data_dir='data/raw/statcast'):
    """
    Calculate all features for a single player.
    
    Parameters:
    -----------
    player_id : int
        MLBAM player ID
    data_dir : str
        Directory where Statcast parquet files are stored
    
    Returns:
    --------
    dict : All calculated features
    """
    from src.data_collection.fetch_statcast import load_player_statcast
    
    # Load data
    data = load_player_statcast(player_id, data_dir)
    
    if data is None or len(data) == 0:
        return None
    
    # Calculate features
    calculator = StatcastFeatureCalculator(data)
    features = calculator.calculate_all_features()
    
    # Add metadata
    features['player_id'] = player_id
    features['total_pitches'] = len(data)
    features['total_pas'] = len(calculator.pas)
    
    return features


def calculate_all_players_features(timeline_csv='data/processed/test_dates.csv',
                                   data_dir='data/raw/statcast',
                                   output_path='data/processed/early_career_stats.csv'):
    """
    Calculate features for all players in timeline.
    
    Parameters:
    -----------
    timeline_csv : str
        Path to timeline CSV with player IDs
    data_dir : str
        Directory with Statcast data
    output_path : str
        Where to save results
    
    Returns:
    --------
    pd.DataFrame : All player features
    """
    timeline = pd.read_csv(timeline_csv)
    timeline = timeline[timeline['success'] == True]
    
    print(f"Calculating features for {len(timeline)} players...")
    
    results = []
    
    for idx, row in timeline.iterrows():
        player_id = int(row['player_id'])
        player_name = row.get('player_name', 'Unknown')
        
        print(f"[{idx+1}/{len(timeline)}] {player_name} (ID: {player_id})...")
        
        features = calculate_player_features(player_id, data_dir)
        
        if features:
            # Add player info from timeline
            features['player_name'] = player_name
            features['debut_date'] = row['debut_date']
            features['date_reached'] = row['date_reached']
            features['n_pa'] = row['n_pa']
            
            results.append(features)
        else:
            print(f"  ✗ No features calculated")
    
    # Convert to DataFrame
    features_df = pd.DataFrame(results)
    
    # Save
    features_df.to_csv(output_path, index=False)
    print(f"\n✓ Saved features for {len(features_df)} players to {output_path}")
    
    return features_df


if __name__ == "__main__":
    # Calculate features for all players
    features_df = calculate_all_players_features()
    
    print("\n" + "="*60)
    print("Feature Summary:")
    print("="*60)
    print(f"Players: {len(features_df)}")
    print(f"Features: {len(features_df.columns)}")
    print("\nSample:")
    print(features_df[['player_name', 'AVG', 'OBP', 'SLG', 'xwOBA', 'xwRC+', 'EV', 'Barrel%']].head())