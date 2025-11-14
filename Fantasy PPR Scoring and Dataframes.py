# === Fantasy PPR Scoring with nflreadpy loader (Pandas pipeline) ===
# - Loads each season via nflreadpy (Polars) → converts to pandas
# - Applies filters & ESPN-standard PPR scoring
# - Includes historical team normalization and week caps by era
# - Produces multi-season leaders/weekly (players), D/ST, and kicker tables

import numpy as np
import pandas as pd
import nflreadpy as nfl

# ----------------------------------
# Config
# ----------------------------------
ALL_SEASONS = list(range(1999, 2025))  # 1999–2024 in one pass

def fantasy_week_max(season: int) -> int:
    # Common fantasy policy: stop before NFL's final week
    # <=2020: Week 16; >=2021: Week 17
    return 16 if season <= 2020 else 17

# ----------------------------------
# Team normalization (historical)
# ----------------------------------
# Normalize to SEASON-ACCURATE abbreviations:
# - Rams:   STL through 2015, LAR 2016+
# - Raiders: OAK through 2019, LV 2020+
# - Chargers: SD through 2016, LAC 2017+
def normalize_team(team: str, season: int) -> str:
    t = (team or "").upper()
    # Rams
    if season <= 2015 and t in {"LAR", "LA", "STL"}:
        return "STL"
    if season >= 2016 and t in {"LAR", "LA", "STL"}:
        return "LAR"
    # Raiders
    if season <= 2019 and t in {"OAK", "LV"}:
        return "OAK"
    if season >= 2020 and t in {"OAK", "LV"}:
        return "LV"
    # Chargers
    if season <= 2016 and t in {"SD", "LAC"}:
        return "SD"
    if season >= 2017 and t in {"SD", "LAC"}:
        return "LAC"
    return t

# ----------------------------------
# Utilities
# ----------------------------------
def base_filter(df: pd.DataFrame, week_min: int, week_max: int) -> pd.DataFrame:
    out = df.query("season_type == 'REG' and @week_min <= week <= @week_max").copy()
    if "play_deleted" in out.columns:
        out = out[out["play_deleted"].fillna(0) != 1]
    if "play_type" in out.columns:
        out = out[out["play_type"] != "no_play"]
    return out

def safe_col(df, name, default=0):
    return df[name].fillna(0) if name in df.columns else 0

# ----------------------------------
# 1) Player PPR (ESPN-standard)
# ----------------------------------
def player_ppr(pbp_reg: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    df = pbp_reg.copy()

    # Flags
    pass_attempt  = (df.get("pass_attempt", df.get("pass", 0)).fillna(0).astype(int) == 1)
    rush_attempt  = (df.get("rush_attempt", df.get("rush", 0)).fillna(0).astype(int) == 1)
    complete_pass = (df.get("complete_pass", 0).fillna(0).astype(int) == 1)

    # -------------------------------
    # Passing → credit the PASSER
    # -------------------------------
    pass_pts = (
        safe_col(df, "passing_yards")*0.04 +
        safe_col(df, "pass_touchdown")*4 -
        safe_col(df, "interception")*2
    )
    passing = pd.DataFrame({
        "season": df["season"],
        "week": df["week"],
        "player_id": df.get("passer_player_id"),
        "player_name": df.get("passer_player_name"),
        "fp": np.where(pass_attempt, pass_pts, 0.0)
    })

    # -------------------------------
    # Rushing → credit the RUSHER
    # -------------------------------
    rush_pts = (
        safe_col(df, "rushing_yards")*0.1 +
        safe_col(df, "rush_touchdown")*6
    )
    rushing = pd.DataFrame({
        "season": df["season"],
        "week": df["week"],
        "player_id": df.get("rusher_player_id"),
        "player_name": df.get("rusher_player_name"),
        "fp": np.where(rush_attempt, rush_pts, 0.0)
    })

    # -----------------------------------------
    # Receiving (PPR) → credit the RECEIVER
    # -----------------------------------------
    recv_pts = (
        1.0 +
        safe_col(df, "receiving_yards")*0.1 +
        safe_col(df, "pass_touchdown")*6
    )
    receiving = pd.DataFrame({
        "season": df["season"],
        "week": df["week"],
        "player_id": df.get("receiver_player_id"),
        "player_name": df.get("receiver_player_name"),
        "fp": np.where(complete_pass, recv_pts, 0.0)
    })

    # ------------------------------------------------------------
    # Two-point conversions
    #   - credit RECEIVER for successful pass 2-pt
    #   - credit RUSHER   for successful rush 2-pt
    # ------------------------------------------------------------
    two_pt_success = (safe_col(df, "two_point_attempt") == 1) & (df.get("two_point_conv_result", "") == "success")

    two_pt_recv = pd.DataFrame({
        "season": df["season"], "week": df["week"],
        "player_id": df.get("receiver_player_id"),
        "player_name": df.get("receiver_player_name"),
        "fp": np.where(two_pt_success & (safe_col(df, "pass") == 1), 2.0, 0.0)
    })

    two_pt_rush = pd.DataFrame({
        "season": df["season"], "week": df["week"],
        "player_id": df.get("rusher_player_id"),
        "player_name": df.get("rusher_player_name"),
        "fp": np.where(two_pt_success & (safe_col(df, "rush") == 1), 2.0, 0.0)
    })

    # ------------------------------------------------------------
    # Fumbles lost → credit the fumbling player (-2)
    # ------------------------------------------------------------
    fum_lost = pd.DataFrame({
        "season": df["season"], "week": df["week"],
        "player_id": df.get("fumbled_1_player_id"),
        "player_name": None, 
        "fp": np.where(safe_col(df, "fumble_lost") == 1, -2.0, 0.0)
    })

    # Stack all role-attributed rows
    parts = [passing, rushing, receiving, two_pt_recv, two_pt_rush, fum_lost]
    all_rows = pd.concat(parts, ignore_index=True)

    # Keep only rows with a valid player_id
    all_rows = all_rows[all_rows["player_id"].notna()].copy()

    # Aggregate to weekly, then season leaders
    weekly = (all_rows.groupby(["season","player_id","player_name","week"], as_index=False)["fp"].sum()
                        .sort_values(["season","player_name","week"]))
    leaders = (weekly.groupby(["season","player_id","player_name"], as_index=False)["fp"].sum()
                      .sort_values(["season","fp"], ascending=[True, False]))

    # Games = count of weeks with non-zero FP
    games = (weekly.loc[weekly["fp"] != 0]
                   .groupby(["season","player_id","player_name"], as_index=False)["week"].nunique()
                   .rename(columns={"week":"games"}))
    leaders = leaders.merge(games, on=["season","player_id","player_name"], how="left")
    leaders["ppg"] = leaders["fp"] / leaders["games"]

    # Rename to canonical keys expected downstream
    leaders = leaders.rename(columns={"player_id":"fantasy_player_id","player_name":"fantasy_player_name"})
    weekly  = weekly.rename(columns={"player_id":"fantasy_player_id","player_name":"fantasy_player_name"})

    return leaders, weekly

# ----------------------------------
# 2) Defense/Special Teams (D/ST)
# ----------------------------------
def dst_scoring(pbp_reg: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    df = pbp_reg.copy()

    # Final scores + teams per game
    scores = (df.groupby(["season","game_id"], as_index=False)[["total_home_score","total_away_score","home_team","away_team"]]
                .agg({"total_home_score":"max","total_away_score":"max","home_team":"first","away_team":"first"})
                .rename(columns={"total_home_score":"home_pts_final","total_away_score":"away_pts_final"}))

    home_rows = scores[["season","game_id","home_team","away_pts_final"]].rename(
        columns={"home_team":"team","away_pts_final":"points_allowed"})
    away_rows = scores[["season","game_id","away_team","home_pts_final"]].rename(
        columns={"away_team":"team","home_pts_final":"points_allowed"})
    pa_tbl = pd.concat([home_rows, away_rows], ignore_index=True)

    # Points-allowed tiers (ESPN-ish)
    def pa_to_pts(pa: int) -> int:
        if pa == 0: return 10
        if pa <= 6: return 7
        if pa <= 13: return 4
        if pa <= 17: return 1
        if pa <= 27: return 0
        if pa <= 34: return -1
        if pa <= 45: return -4
        return -5

    pa_tbl["pa_pts"] = pa_tbl["points_allowed"].astype(int).map(pa_to_pts)

    # Tallies by defense
    by_def = (df.groupby(["season","game_id","defteam"], as_index=False)
                .agg({"sack":"sum","interception":"sum","safety":"sum","punt_blocked":"sum"})
                .rename(columns={"defteam":"team"}))

    # Blocked FGs
    tmp = df.copy()
    tmp["fg_block"] = (tmp.get("field_goal_result","").astype(str).str.lower() == "blocked").astype(int)
    by_def_fgblk = (tmp.groupby(["season","game_id","defteam"], as_index=False)["fg_block"].sum()
                      .rename(columns={"defteam":"team"}))

    # Fumble recoveries credited to recovering team
    fr_parts = []
    if "fumble_recovery_1_team" in df.columns:
        fr1 = df[df["fumble_recovery_1_team"].notna()][["season","game_id","fumble_recovery_1_team"]].copy()
        fr1["team"] = fr1["fumble_recovery_1_team"]; fr1["fr"] = 1
        fr_parts.append(fr1[["season","game_id","team","fr"]])
    if "fumble_recovery_2_team" in df.columns:
        fr2 = df[df["fumble_recovery_2_team"].notna()][["season","game_id","fumble_recovery_2_team"]].copy()
        fr2["team"] = fr2["fumble_recovery_2_team"]; fr2["fr"] = 1
        fr_parts.append(fr2[["season","game_id","team","fr"]])
    by_fr = (pd.concat(fr_parts, ignore_index=True)
             .groupby(["season","game_id","team"], as_index=False)["fr"].sum()) if fr_parts else \
            pd.DataFrame(columns=["season","game_id","team","fr"])

    # Return TDs (credited to return team)
    ret = df[(safe_col(df,"return_touchdown") == 1)]
    by_ret_td = (ret.groupby(["season","game_id","return_team"], as_index=False).size()
                   .rename(columns={"return_team":"team","size":"ret_td"}))

    # Merge all components
    dst = pa_tbl[["season","game_id","team","pa_pts"]].copy()
    for part in [
        by_def[["season","game_id","team","sack","interception","safety","punt_blocked"]],
        by_def_fgblk[["season","game_id","team","fg_block"]],
        by_fr[["season","game_id","team","fr"]],
        by_ret_td[["season","game_id","team","ret_td"]],
    ]:
        dst = dst.merge(part, on=["season","game_id","team"], how="left")

    dst = dst.fillna(0)
    for c in ["sack","interception","safety","punt_blocked","fg_block","fr","ret_td"]:
        if c in dst.columns: dst[c] = dst[c].astype(int)

    # Normalize historical team code
    dst["team"] = [normalize_team(t, s) for t, s in zip(dst["team"].astype(str), dst["season"].astype(int))]

    dst["dst_fp"] = (
        dst["sack"]*1 + dst["interception"]*2 + dst["fr"]*2 + dst["safety"]*2
        + dst["ret_td"]*6 + (dst["punt_blocked"] + dst["fg_block"])*2 + dst["pa_pts"]
    )

    per_game = dst.sort_values(["season","game_id","team"]).reset_index(drop=True)
    season = (per_game.groupby(["season","team"], as_index=False)["dst_fp"].sum()
                      .sort_values(["season","dst_fp"], ascending=[True, False]))
    return season, per_game

# ----------------------------------
# 3) Kickers (ESPN-style)
# ----------------------------------
def kicker_scoring(pbp_reg: pd.DataFrame, miss_pat_minus_one=False) -> tuple[pd.DataFrame, pd.DataFrame]:
    df = pbp_reg.copy()
    kick = df[(safe_col(df, "field_goal_attempt") == 1) | (safe_col(df, "extra_point_attempt") == 1)].copy()

    for c in ["field_goal_result","extra_point_result"]:
        if c in kick.columns: 
            kick[c] = kick[c].astype(str).str.lower()

    # FG made by distance
    fg_made = (safe_col(kick, "field_goal_attempt") == 1) & (kick["field_goal_result"] == "made")
    dist = safe_col(kick, "kick_distance").astype(float)
    fg_pts = np.where(fg_made & (dist >= 60), 6,
              np.where(fg_made & (dist >= 50), 5,
              np.where(fg_made & (dist >= 40), 4,
              np.where(fg_made, 3, 0))))

    # Missed FG (includes blocked)
    fg_miss = (safe_col(kick, "field_goal_attempt") == 1) & (kick["field_goal_result"].isin(["missed","blocked"]))
    fg_miss_pts = np.where(fg_miss, -1, 0)

    # PATs
    pat_attempt = (safe_col(kick, "extra_point_attempt") == 1)
    pat_good = pat_attempt & (kick["extra_point_result"].isin(["good","made"]))
    pat_pts = np.where(pat_good, 1, 0)
    pat_missed = pat_attempt & (kick["extra_point_result"].isin(["failed","missed","blocked"]))
    pat_miss_pts = np.where(miss_pat_minus_one & pat_missed, -1, 0)

    kick["k_fp"] = fg_pts + fg_miss_pts + pat_pts + pat_miss_pts

    id_col = "kicker_player_id" if "kicker_player_id" in kick.columns else "fantasy_player_id"
    name_col = "kicker_player_name" if "kicker_player_name" in kick.columns else "fantasy_player_name"

    per_game = (kick.groupby(["season","game_id", id_col, name_col, "posteam"], as_index=False)
                  .agg(k_fp=("k_fp","sum"),
                       fg_made=("field_goal_attempt", lambda s: int(((s==1) & (kick.loc[s.index,'field_goal_result'].eq('made'))).sum())),
                       fg_att=("field_goal_attempt","sum"),
                       pat_made=("extra_point_attempt", lambda s: int(((s==1) & (kick.loc[s.index,'extra_point_result'].isin(['good','made']))).sum())),
                       pat_att=("extra_point_attempt","sum"))
               )

    # Normalize team code historically for posteam in outputs
    per_game["posteam"] = [normalize_team(t, s) for t, s in zip(per_game["posteam"].astype(str), per_game["season"].astype(int))]

    season = (per_game.groupby(["season", id_col, name_col, "posteam"], as_index=False)
                      .agg(total_fp=("k_fp","sum"),
                           games=("game_id","nunique"),
                           fg_made=("fg_made","sum"),
                           fg_att=("fg_att","sum"),
                           pat_made=("pat_made","sum"),
                           pat_att=("pat_att","sum")))
    season["ppg"] = season["total_fp"] / season["games"]
    season = season.sort_values(["season","total_fp","ppg"], ascending=[True, False, False])
    return season, per_game

# ----------------------------------
# Runner (streams seasons safely) — uses nflreadpy loader
# ----------------------------------
players_all, players_weekly_all = [], []
dst_all, dst_games_all = [], []
k_all, k_games_all = [], []

for yr in ALL_SEASONS:
    print(f"Processing {yr} ...")
    # Load via nflreadpy (Polars) then convert to pandas
    pbp_pl = nfl.load_pbp([yr])      # Polars DataFrame
    pbp_y  = pbp_pl.to_pandas()      # hand off to pandas pipeline

    wk_max = fantasy_week_max(yr)
    pbp_reg = base_filter(pbp_y, week_min=1, week_max=wk_max)

    # Optional RAM saver
    for c in pbp_reg.select_dtypes("float64").columns:
        pbp_reg[c] = pbp_reg[c].astype("float32")
    print(f"{yr} done.\nDowncasting floats.")

    # Players
    p_season, p_weekly = player_ppr(pbp_reg)
    players_all.append(p_season)
    players_weekly_all.append(p_weekly)

    # D/ST
    d_season, d_games = dst_scoring(pbp_reg)
    dst_all.append(d_season)
    dst_games_all.append(d_games)

    # Kickers
    k_season, k_games = kicker_scoring(pbp_reg, miss_pat_minus_one=False)
    k_all.append(k_season)
    k_games_all.append(k_games)

# Concatenate multi-season outputs
players_leaders_multi = pd.concat(players_all, ignore_index=True)
players_weekly_multi  = pd.concat(players_weekly_all, ignore_index=True)
dst_season_multi      = pd.concat(dst_all, ignore_index=True)
dst_per_game_multi    = pd.concat(dst_games_all, ignore_index=True)
k_season_multi        = pd.concat(k_all, ignore_index=True)
k_per_game_multi      = pd.concat(k_games_all, ignore_index=True)

# ---- Canonicalize player-season leaders (dedupe to 1 row per season+player) ----
players_leaders_multi = (
    players_leaders_multi
      .sort_values(['season','fantasy_player_id','fp'], ascending=[True, True, False])
      .groupby(['season','fantasy_player_id'], as_index=False)
      .agg(
          fp=('fp','first'),                 # season total FP
          games=('games','max'),             # keep max credited games if duplicates disagree
          fantasy_player_name=('fantasy_player_name','first')
      )
)
# Guarantee uniqueness
assert not players_leaders_multi.duplicated(['season','fantasy_player_id']).any(), "Duplicate player-season keys remain."

# ================================
# QB dataframe construction (final, PPR-aligned games, position-pure)
# Reuses from PPR script:
#   - ALL_SEASONS
#   - fantasy_week_max(season)
#   - base_filter(df, week_min, week_max)
#   - safe_col(df, name, default=0)
#   - normalize_team(team, season)
#   - players_leaders_multi with unique ['season','fantasy_player_id'] and cols ['fp','games']
#   - import nflreadpy as nfl
# ================================

# ---- positions helper (nflreadpy rosters) ----
def _load_positions_from_rosters(seasons) -> pd.DataFrame:
    """
    Load (season, gsis_id, position) from nflreadpy rosters and collapse to one row per season+player.
    Merges on GSIS ID which matches your fantasy_player_id (e.g. '00-0033873').
    Coerces seasons to plain Python ints to satisfy nflreadpy's type checks.
    """
    # Coerce seasons -> clean list[int]
    season_series = pd.Series(seasons)
    seasons_clean = (
        season_series.dropna()
        .astype(int)                      
        .astype(object)                   
        .apply(int)
        .drop_duplicates()
        .tolist()
    )

    rosters_pl = nfl.load_rosters(seasons_clean)
    rosters = rosters_pl.to_pandas()

    cand_id_cols = [c for c in ["gsis_id", "player_id", "gsis_player_id"] if c in rosters.columns]
    if not cand_id_cols:
        raise ValueError("No GSIS id-like column found in rosters. Expected one of: gsis_id/player_id/gsis_player_id.")
    id_col = cand_id_cols[0]

    need = {"season", id_col, "position"}
    if not need.issubset(rosters.columns):
        missing = need - set(rosters.columns)
        raise ValueError(f"Rosters table missing required columns: {missing}")

    pos_tbl = (
        rosters[["season", id_col, "position"]]
        .dropna(subset=[id_col])
        .copy()
    )
    pos_tbl["position"] = pos_tbl["position"].astype(str).str.upper()
    pos_tbl = (
        pos_tbl.sort_values(["season", id_col, "position"])
               .groupby(["season", id_col], as_index=False)
               .agg(position=("position", "first"))
    )
    pos_tbl = pos_tbl.rename(columns={id_col: "fantasy_player_id"})
    return pos_tbl


def _ensure_qb_cols(df: pd.DataFrame) -> pd.DataFrame:
    """Make sure QB/stat columns exist and are numeric so groupby/agg works on older seasons."""
    must_int = [
        "complete_pass", "pass_touchdown", "interception", "sack",
        "rush_touchdown", "qb_scramble"
    ]
    must_float = [
        "passing_yards", "air_yards", "yards_after_catch",
        "epa", "rushing_yards"
    ]
    
    if "pass_attempt" not in df.columns and "pass" in df.columns:
        df["pass_attempt"] = df["pass"]
    if "rush_attempt" not in df.columns and "rush" in df.columns:
        df["rush_attempt"] = df["rush"]

    for c in must_int:
        if c not in df.columns:
            df[c] = 0
        df[c] = df[c].fillna(0).astype("int64")

    for c in must_float:
        if c not in df.columns:
            df[c] = 0.0
        df[c] = df[c].fillna(0.0).astype("float64")

    for c in ["pass_attempt", "rush_attempt"]:
        if c not in df.columns:
            df[c] = 0
        df[c] = df[c].fillna(0).astype("int64")

    return df


def engineer_qb_features_for_season(season: int) -> pd.DataFrame:
    pbp_pl = nfl.load_pbp([season])
    pbp = pbp_pl.to_pandas()
    wk_max = fantasy_week_max(season)
    pbp = base_filter(pbp, week_min=1, week_max=wk_max)

    pbp = _ensure_qb_cols(pbp)

    # Role flags
    pass_attempt_flag = (pbp.get('pass_attempt', pbp.get('pass', 0)).fillna(0).astype(int) == 1)
    rush_attempt_flag = (pbp.get('rush_attempt', pbp.get('rush', 0)).fillna(0).astype(int) == 1)

    # Use actor IDs (passer/rusher)
    pbp['player_id'] = np.where(pass_attempt_flag, pbp.get('passer_player_id'),
                        np.where(rush_attempt_flag, pbp.get('rusher_player_id'), np.nan))
    pbp['player_name'] = np.where(pass_attempt_flag, pbp.get('passer_player_name'),
                          np.where(rush_attempt_flag, pbp.get('rusher_player_name'), np.nan))
    pbp = pbp[pbp['player_id'].notna()].copy()

    pbp['__pass_'] = pass_attempt_flag.astype(int)
    pbp['__rush_'] = rush_attempt_flag.astype(int)

    # Season aggregates per player
    g = (
        pbp.groupby(['season','player_id','player_name'])
        .agg(
            games=('week', 'nunique'),             
            plays=('play_id', 'count'),
            pass_plays=('__pass_', 'sum'),
            rush_plays=('__rush_', 'sum'),
            comp=('complete_pass', 'sum'),
            pass_yards=('passing_yards', 'sum'),
            pass_td=('pass_touchdown', 'sum'),
            interceptions=('interception', 'sum'),
            sacks=('sack', 'sum'),
            air_yards=('air_yards', 'sum'),
            yac=('yards_after_catch', 'sum'),
            epa_sum=('epa', 'sum'),
            rush_yards=('rushing_yards', 'sum'),
            rush_td=('rush_touchdown', 'sum'),
            scrambles=('qb_scramble', 'sum')
        )
        .reset_index()
    )

    team_mode = (
        pbp.groupby(['season','player_id'])['posteam']
        .agg(lambda s: s.dropna().astype(str).mode().iloc[0] if len(s.dropna()) > 0 else np.nan)
        .rename('team')
        .reset_index()
    )
    g = g.merge(team_mode, on=['season','player_id'], how='left')
    if 'team' in g.columns:
        g['team'] = [normalize_team(str(t), int(s)) if pd.notna(t) else np.nan
                     for t, s in zip(g['team'], g['season'])]

    g['comp_pct'] = np.where(g['pass_plays'] > 0, g['comp'] / g['pass_plays'], 0.0)
    g['ypa'] = np.where(g['pass_plays'] > 0, g['pass_yards'] / g['pass_plays'], 0.0)
    g['td_rate'] = np.where(g['pass_plays'] > 0, g['pass_td'] / g['pass_plays'], 0.0)
    g['int_rate'] = np.where(g['pass_plays'] > 0, g['interceptions'] / g['pass_plays'], 0.0)
    g['sack_rate'] = np.where((g['pass_plays'] + g['sacks']) > 0, g['sacks'] / (g['pass_plays'] + g['sacks']), 0.0)
    g['air_ypa'] = np.where(g['pass_plays'] > 0, g['air_yards'] / g['pass_plays'], 0.0)
    g['yac_ypa'] = np.where(g['pass_plays'] > 0, g['yac'] / g['pass_plays'], 0.0)
    g['epa_per_pass'] = np.where(g['pass_plays'] > 0, g['epa_sum'] / g['pass_plays'], 0.0)
    g['rush_yards_per_att'] = np.where(g['rush_plays'] > 0, g['rush_yards'] / g['rush_plays'], 0.0)

    denom = g['pass_plays'] + g['rush_plays']
    g['passer_share'] = np.where(denom > 0, g['pass_plays'] / denom, 0.0)

    g = g.rename(columns={'player_id': 'fantasy_player_id', 'player_name': 'fantasy_player_name'})
    g = g.drop(columns=[c for c in g.columns if c.startswith('__')], errors='ignore')
    return g


def build_qb_features(seasons):
    frames = []
    for yr in seasons:
        print(f"Engineering QB features for {yr} ...")
        frames.append(engineer_qb_features_for_season(yr))
    out = pd.concat(frames, ignore_index=True)

    # Real quarterbacks only (pre-filter by usage/volume)
    out = out[
        (out['pass_plays'] >= 150) |
        ((out['pass_plays'] >= 100) & (out['passer_share'] >= 0.60))
    ].copy()
    return out


def merge_qb_with_targets(qb_feats: pd.DataFrame, players_leaders_multi: pd.DataFrame) -> pd.DataFrame:
    needed = {'season','fantasy_player_id','fp','games'}
    missing = needed - set(players_leaders_multi.columns)
    if missing:
        raise ValueError(f"players_leaders_multi missing columns: {missing}")
    assert not players_leaders_multi.duplicated(['season','fantasy_player_id']).any(), \
        "players_leaders_multi must be unique per season+player (dedupe earlier)."

    # ---- bring in positions from nflreadpy rosters and attach to qb_feats ----
    positions_tbl = _load_positions_from_rosters(qb_feats["season"].unique())
    qb_feats = qb_feats.merge(
        positions_tbl, on=["season","fantasy_player_id"], how="left", validate="many_to_one"
    )

    # Merge FP and PPR-defined games; replace provisional games with PPR games
    qb = qb_feats.drop(columns=['games'], errors='ignore').merge(
        players_leaders_multi[['season','fantasy_player_id','fp','games']].rename(columns={'games':'games_ppr'}),
        on=['season','fantasy_player_id'], how='left', validate='many_to_one'
    )
    qb['fp'] = qb['fp'].fillna(0.0)
    qb = qb.rename(columns={'games_ppr':'games'})

    # Hard filter to true QBs
    pre_n = len(qb)
    qb = qb[qb['position'].str.upper().eq('QB')].copy()
    post_n = len(qb)
    print(f"QB position filter: kept {post_n}/{pre_n} rows where position == 'QB'")

    # Stability filter
    qb = qb[qb['games'] >= 4].copy()
    return qb

qb_feats = build_qb_features(ALL_SEASONS)
qb_df = merge_qb_with_targets(qb_feats, players_leaders_multi)

# ================================
# WR dataframe construction (final, PPR-aligned games, roster-position filtered)
# Reuses from PPR script:
#   - ALL_SEASONS
#   - fantasy_week_max(season)
#   - base_filter(df, week_min, week_max)
#   - normalize_team(team, season)
#   - players_leaders_multi (unique ['season','fantasy_player_id'] with ['fp','games','fantasy_player_name'])
#   - import nflreadpy as nfl
# ================================

import numpy as np
import pandas as pd
from functools import lru_cache

# ---------- helpers to attach roster positions ----------
def _to_py_int_list(values) -> list[int]:
    s = pd.Series(values)
    return (
        s.dropna()
         .astype(int).astype(object).apply(int)
         .drop_duplicates()
         .tolist()
    )

@lru_cache(maxsize=1)
def load_positions_for(seasons_key: tuple[int, ...]) -> pd.DataFrame:
    """
    Cached loader of (season, fantasy_player_id, position) using nflreadpy rosters.
    Pass seasons as a sorted tuple of ints: tuple(sorted(unique_seasons)).
    """
    seasons = list(seasons_key)
    rosters_pl = nfl.load_rosters(seasons) 
    rosters = rosters_pl.to_pandas()

    cand_id_cols = [c for c in ["gsis_id", "player_id", "gsis_player_id"] if c in rosters.columns]
    if not cand_id_cols:
        raise ValueError("Rosters missing GSIS id column. Expected one of gsis_id/player_id/gsis_player_id.")
    id_col = cand_id_cols[0]

    pos_tbl = (
        rosters[["season", id_col, "position"]]
        .dropna(subset=[id_col])
        .copy()
    )
    pos_tbl["position"] = pos_tbl["position"].astype(str).str.upper().str.strip()
    pos_tbl = (
        pos_tbl.sort_values(["season", id_col, "position"])
               .groupby(["season", id_col], as_index=False)
               .agg(position=("position", "first"))
               .rename(columns={id_col: "fantasy_player_id"})
    )
    return pos_tbl

# ---------- WR feature engineering ----------
def _ensure_wr_cols(df: pd.DataFrame) -> pd.DataFrame:
    """Ensure columns exist/cast for older seasons so WR aggregations are stable."""
    must_int = [
        "complete_pass", "pass_touchdown", "rush_touchdown",
        "pass_attempt", "rush_attempt"
    ]
    must_float = [
        "receiving_yards", "rushing_yards", "air_yards", "yards_after_catch", "epa"
    ]

    if "pass_attempt" not in df.columns and "pass" in df.columns:
        df["pass_attempt"] = df["pass"]
    if "rush_attempt" not in df.columns and "rush" in df.columns:
        df["rush_attempt"] = df["rush"]

    for c in must_int:
        if c not in df.columns:
            df[c] = 0
        df[c] = df[c].fillna(0).astype("int64")

    for c in must_float:
        if c not in df.columns:
            df[c] = 0.0
        df[c] = df[c].fillna(0.0).astype("float64")

    return df


def engineer_wr_features_for_season(season: int) -> pd.DataFrame:
    pbp_pl = nfl.load_pbp([season])
    pbp = pbp_pl.to_pandas()
    wk_max = fantasy_week_max(season)
    pbp = base_filter(pbp, week_min=1, week_max=wk_max)

    pbp = _ensure_wr_cols(pbp)

    # ---------- Targets / receptions (receiver role on pass attempts) ----------
    is_pass = (pbp["pass_attempt"] == 1)
    has_tgt = pbp["receiver_player_id"].notna()

    tgt_rows = pbp[is_pass & has_tgt].copy()
    rec_rows = pbp[(pbp["complete_pass"] == 1) & has_tgt].copy()

    rec_g = (
        rec_rows.groupby(["season", "receiver_player_id", "receiver_player_name"], as_index=False)
        .agg(
            receptions=("complete_pass", "sum"),
            rec_yards=("receiving_yards", "sum"),
            rec_td=("pass_touchdown", "sum"),
            air_yards=("air_yards", "sum"),
            yac=("yards_after_catch", "sum"),
            rec_plays=("play_id", "count"),
        )
    )

    tgt_g = (
        tgt_rows.groupby(["season", "receiver_player_id", "receiver_player_name"], as_index=False)
        .agg(
            targets=("play_id", "count"),
        )
    )

    # ---------- WR rushing (rusher role) ----------
    is_rush = (pbp["rush_attempt"] == 1)
    rush_rows = pbp[is_rush & pbp["rusher_player_id"].notna()].copy()

    rush_g = (
        rush_rows.groupby(["season", "rusher_player_id", "rusher_player_name"], as_index=False)
        .agg(
            wr_rush_att=("rush_attempt", "sum"),
            wr_rush_yards=("rushing_yards", "sum"),
            wr_rush_td=("rush_touchdown", "sum"),
        )
        .rename(columns={
            "rusher_player_id": "receiver_player_id",
            "rusher_player_name": "receiver_player_name"
        })
    )

    # ---------- Combine receiving + targets + rushing ----------
    wr = (
        tgt_g.merge(rec_g, on=["season","receiver_player_id","receiver_player_name"], how="outer")
             .merge(rush_g, on=["season","receiver_player_id","receiver_player_name"], how="left")
             .fillna({
                 "receptions":0, "rec_yards":0.0, "rec_td":0,
                 "air_yards":0.0, "yac":0.0, "rec_plays":0,
                 "wr_rush_att":0, "wr_rush_yards":0.0, "wr_rush_td":0,
                 "targets":0
             })
    )

    # ---------- Modal team label per player-season ----------
    team_source = pd.concat([
        tgt_rows[["season","receiver_player_id","posteam"]].rename(columns={"receiver_player_id":"pid", "posteam":"team"}),
        rush_rows[["season","rusher_player_id","posteam"]].rename(columns={"rusher_player_id":"pid", "posteam":"team"}),
    ], ignore_index=True)

    team_mode = (
        team_source.dropna(subset=["pid","team"])
                   .groupby(["season","pid"])["team"]
                   .agg(lambda s: s.dropna().astype(str).mode().iloc[0] if len(s.dropna()) > 0 else np.nan)
                   .reset_index()
                   .rename(columns={"pid":"receiver_player_id","team":"team"})
    )

    wr = wr.merge(team_mode, on=["season","receiver_player_id"], how="left")
    if "team" in wr.columns:
        wr["team"] = [normalize_team(str(t), int(s)) if pd.notna(t) else np.nan
                      for t, s in zip(wr["team"], wr["season"])]

    # ---------- Team targets & target share ----------
    team_tgts = (
        tgt_rows.groupby(["season","posteam"], as_index=False)
                .agg(team_targets=("play_id","count"))
                .rename(columns={"posteam":"team_raw"})
    )
    wr = wr.merge(team_tgts, left_on=["season","team"], right_on=["season","team_raw"], how="left")
    wr = wr.drop(columns=["team_raw"], errors="ignore")
    wr["team_targets"] = wr["team_targets"].fillna(0).astype(float)

    # ---------- Feature engineering ----------
    wr["adot"] = np.where(wr["targets"] > 0, wr["air_yards"] / wr["targets"], 0.0)
    wr["ypr"] = np.where(wr["receptions"] > 0, wr["rec_yards"] / wr["receptions"], 0.0)
    wr["ypt"] = np.where(wr["targets"] > 0, wr["rec_yards"] / wr["targets"], 0.0)
    wr["yac_per_rec"] = np.where(wr["receptions"] > 0, wr["yac"] / wr["receptions"], 0.0)
    wr["target_share"] = np.where(wr["team_targets"] > 0, wr["targets"] / wr["team_targets"], 0.0)

    # Provisional "games" = #weeks with involvement (either a target or a rush)
    inv_rows = pd.concat([
        tgt_rows[["season","week","receiver_player_id"]].rename(columns={"receiver_player_id":"pid"}),
        rush_rows[["season","week","rusher_player_id"]].rename(columns={"rusher_player_id":"pid"})
    ], ignore_index=True).dropna(subset=["pid"])

    inv_weeks = (
        inv_rows.groupby(["season","pid"], as_index=False)["week"]
                .nunique()
                .rename(columns={"pid":"receiver_player_id","week":"games"})
    )

    wr = wr.merge(inv_weeks, on=["season","receiver_player_id"], how="left")
    wr["games"] = wr["games"].fillna(0).astype(float)

    # ---------- rename keys to merge with PPR leaders ----------
    wr = wr.rename(columns={
        "receiver_player_id":"fantasy_player_id",
        "receiver_player_name":"fantasy_player_name",
    })

    cols_order = [
        "season","fantasy_player_id","fantasy_player_name","team",
        "targets","receptions","rec_yards","rec_td","air_yards","yac",
        "adot","ypr","ypt","yac_per_rec",
        "team_targets","target_share",
        "wr_rush_att","wr_rush_yards","wr_rush_td",
        "games"
    ]
    wr = wr[[c for c in cols_order if c in wr.columns]]
    return wr


def build_wr_features(seasons):
    frames = []
    for yr in seasons:
        print(f"Engineering WR features for {yr} ...")
        frames.append(engineer_wr_features_for_season(yr))
    out = pd.concat(frames, ignore_index=True)

    out = out[
        (out["targets"] >= 50) |
        ((out["targets"] >= 30) & (out["games"] >= 4))
    ].copy()
    return out


def merge_wr_with_targets(wr_feats: pd.DataFrame, players_leaders_multi: pd.DataFrame) -> pd.DataFrame:
    needed = {"season","fantasy_player_id","fp","games"}
    missing = needed - set(players_leaders_multi.columns)
    if missing:
        raise ValueError(f"players_leaders_multi missing columns: {missing}")
    assert not players_leaders_multi.duplicated(["season","fantasy_player_id"]).any(), \
        "players_leaders_multi must be unique per season+player (dedupe earlier)."

    # --- attach & filter by roster position (WR only) ---
    wr_seasons_key = tuple(sorted(_to_py_int_list(wr_feats["season"].unique())))
    positions_tbl = load_positions_for(wr_seasons_key)

    wr_feats = wr_feats.merge(
        positions_tbl, on=["season","fantasy_player_id"], how="left", validate="many_to_one"
    )
    pre_n = len(wr_feats)
    wr_feats = wr_feats[wr_feats["position"].astype(str).str.upper().eq("WR")].copy()
    post_n = len(wr_feats)
    print(f"WR position filter: kept {post_n}/{pre_n} rows where position == 'WR'")

    # --- merge FP + PPR-defined games ---
    wr = wr_feats.drop(columns=["games"], errors="ignore").merge(
        players_leaders_multi[["season","fantasy_player_id","fp","games"]]
            .rename(columns={"games":"games_ppr"}),
        on=["season","fantasy_player_id"],
        how="left",
        validate="many_to_one"
    )
    wr["fp"] = wr["fp"].fillna(0.0)
    wr = wr.rename(columns={"games_ppr":"games"})

    wr = wr[wr["games"] >= 4].copy()
    return wr

wr_feats = build_wr_features(ALL_SEASONS)
wr_df = merge_wr_with_targets(wr_feats, players_leaders_multi)

# ================================
# RB dataframe construction (final, PPR-aligned games, position-filtered)
# Reuses from PPR script:
#   - ALL_SEASONS
#   - fantasy_week_max(season)
#   - base_filter(df, week_min, week_max)
#   - safe_col(df, name, default=0)
#   - normalize_team(team, season)
#   - players_leaders_multi with unique ['season','fantasy_player_id'] and cols ['fp','games','fantasy_player_name']
#   - import nflreadpy as nfl
# ================================
from functools import lru_cache
import numpy as np
import pandas as pd

# ---------- Roster positions helper ----------
def _to_py_int_list(arr_like) -> list[int]:
    return [int(x) for x in pd.Series(arr_like).dropna().unique().tolist()]

@lru_cache(maxsize=None)
def load_positions_for(seasons_key: tuple[int, ...]) -> pd.DataFrame:
    """
    Load (season, gsis_id, position) from nflreadpy rosters and collapse to one row per season+player.
    Merges on GSIS ID which matches your fantasy_player_id (e.g. '00-0033873').
    """
    seasons = list(seasons_key)
    rosters_pl = nfl.load_rosters(seasons)
    rosters = rosters_pl.to_pandas()

    id_col = None
    for cand in ["gsis_id", "player_id", "nfl_id", "gsis"]:
        if cand in rosters.columns:
            id_col = cand
            break
    if id_col is None:
        raise KeyError("Could not find a GSIS ID column in rosters. Expected one of ['gsis_id','player_id','nfl_id','gsis'].")

    keep = rosters[["season", id_col, "position"]].copy()
    keep = keep.rename(columns={id_col: "fantasy_player_id"})
    keep["fantasy_player_id"] = keep["fantasy_player_id"].astype(str)

    # Collapse to one row per season+player
    keep["position"] = keep["position"].astype(str).str.upper().replace({"NONE": np.nan})
    positions_tbl = (
        keep.dropna(subset=["fantasy_player_id"])
            .sort_values(["season", "fantasy_player_id"])
            .groupby(["season", "fantasy_player_id"], as_index=False)
            .agg(position=("position", "first"))
    )
    return positions_tbl


def _ensure_rb_cols(df: pd.DataFrame) -> pd.DataFrame:
    """Ensure columns exist/cast across vintages for stable RB aggregations."""
    # Integers / flags
    must_int = [
        "pass_attempt", "rush_attempt", "complete_pass",
        "rush_touchdown", "pass_touchdown",
        "fumble_lost",
    ]
    # Floats / totals
    must_float = [
        "rushing_yards", "receiving_yards", "air_yards", "yards_after_catch", "epa",
        "yardline_100"
    ]

    if "pass_attempt" not in df.columns and "pass" in df.columns:
        df["pass_attempt"] = df["pass"]
    if "rush_attempt" not in df.columns and "rush" in df.columns:
        df["rush_attempt"] = df["rush"]

    for c in must_int:
        if c not in df.columns:
            df[c] = 0
        df[c] = df[c].fillna(0).astype("int64")

    for c in must_float:
        if c not in df.columns:
            df[c] = 0.0
        df[c] = df[c].astype("float64").fillna(0.0)

    return df


def engineer_rb_features_for_season(season: int) -> pd.DataFrame:
    pbp_pl = nfl.load_pbp([season])
    pbp = pbp_pl.to_pandas()
    wk_max = fantasy_week_max(season)
    pbp = base_filter(pbp, week_min=1, week_max=wk_max)

    pbp = _ensure_rb_cols(pbp)

    # ---------- RUSHING (rusher role) ----------
    rush_rows = pbp[(pbp["rush_attempt"] == 1) & pbp["rusher_player_id"].notna()].copy()

    # Goal-line context
    rush_rows["gl_att"] = (rush_rows["yardline_100"] <= 5).astype(int)
    rush_rows["i10_att"] = (rush_rows["yardline_100"] <= 10).astype(int)

    # Fumbles lost credited to the rusher
    rush_rows["fl_rush"] = (
        (rush_rows["fumble_lost"] == 1) &
        (rush_rows["rusher_player_id"] == rush_rows.get("fumbled_1_player_id"))
    ).astype(int)

    rush_g = (
        rush_rows.groupby(["season", "rusher_player_id", "rusher_player_name"], as_index=False)
        .agg(
            carries=("rush_attempt", "sum"),
            rush_yards=("rushing_yards", "sum"),
            rush_td=("rush_touchdown", "sum"),
            rush_epa_sum=("epa", "sum"),
            gl_carries=("gl_att", "sum"),
            i10_carries=("i10_att", "sum"),
            fumbles_lost_rush=("fl_rush", "sum")
        )
        .rename(columns={
            "rusher_player_id":"player_id",
            "rusher_player_name":"player_name"
        })
    )

    # ---------- RECEIVING (receiver role) ----------
    tgt_rows = pbp[(pbp["pass_attempt"] == 1) & pbp["receiver_player_id"].notna()].copy()
    rec_rows = pbp[(pbp["complete_pass"] == 1) & pbp["receiver_player_id"].notna()].copy()

    # Fumbles lost credited to the receiver
    rec_rows["fl_rec"] = (
        (rec_rows["fumble_lost"] == 1) &
        (rec_rows["receiver_player_id"] == rec_rows.get("fumbled_1_player_id"))
    ).astype(int)

    tgt_g = (
        tgt_rows.groupby(["season","receiver_player_id","receiver_player_name"], as_index=False)
        .agg(targets=("play_id","count"))
        .rename(columns={"receiver_player_id":"player_id","receiver_player_name":"player_name"})
    )

    rec_g = (
        rec_rows.groupby(["season","receiver_player_id","receiver_player_name"], as_index=False)
        .agg(
            receptions=("complete_pass","sum"),
            rec_yards=("receiving_yards","sum"),
            rec_td=("pass_touchdown","sum"),
            air_yards=("air_yards","sum"),
            yac=("yards_after_catch","sum"),
            rec_epa_sum=("epa","sum"),
            fumbles_lost_rec=("fl_rec","sum")
        )
        .rename(columns={"receiver_player_id":"player_id","receiver_player_name":"player_name"})
    )

    # ---------- Combine rushing + receiving ----------
    rb = (
        rush_g.merge(tgt_g, on=["season","player_id","player_name"], how="outer")
              .merge(rec_g, on=["season","player_id","player_name"], how="outer")
              .fillna({
                  "carries":0, "rush_yards":0.0, "rush_td":0,
                  "rush_epa_sum":0.0, "gl_carries":0, "i10_carries":0, "fumbles_lost_rush":0,
                  "targets":0, "receptions":0, "rec_yards":0.0, "rec_td":0,
                  "air_yards":0.0, "yac":0.0, "rec_epa_sum":0.0, "fumbles_lost_rec":0
              })
    )

    # ---------- Modal team per player-season (use both rush & target rows) ----------
    team_source = pd.concat([
        rush_rows[["season","rusher_player_id","posteam"]].rename(columns={"rusher_player_id":"pid","posteam":"team"}),
        tgt_rows[["season","receiver_player_id","posteam"]].rename(columns={"receiver_player_id":"pid","posteam":"team"})
    ], ignore_index=True)

    team_mode = (
        team_source.dropna(subset=["pid","team"])
                   .groupby(["season","pid"])["team"]
                   .agg(lambda s: s.dropna().astype(str).mode().iloc[0] if len(s.dropna()) > 0 else np.nan)
                   .reset_index()
                   .rename(columns={"pid":"player_id","team":"team"})
    )

    rb = rb.merge(team_mode, on=["season","player_id"], how="left")
    if "team" in rb.columns:
        rb["team"] = [normalize_team(str(t), int(s)) if pd.notna(t) else np.nan
                      for t, s in zip(rb["team"], rb["season"])]

    # ---------- Team context: team carries & team targets ----------
    team_carries = (
        rush_rows.groupby(["season","posteam"], as_index=False)
                 .agg(team_carries=("rush_attempt","sum"))
                 .rename(columns={"posteam":"team_raw"})
    )
    team_targets = (
        tgt_rows.groupby(["season","posteam"], as_index=False)
                .agg(team_targets=("play_id","count"))
                .rename(columns={"posteam":"team_raw"})
    )

    rb = rb.merge(team_carries, left_on=["season","team"], right_on=["season","team_raw"], how="left")
    rb = rb.merge(team_targets, left_on=["season","team"], right_on=["season","team_raw"], how="left")
    rb = rb.drop(columns=["team_raw"], errors="ignore")
    rb["team_carries"] = rb["team_carries"].fillna(0).astype(float)
    rb["team_targets"] = rb["team_targets"].fillna(0).astype(float)

    # ---------- Derived features ----------
    # Efficiency
    rb["ypc"] = np.where(rb["carries"] > 0, rb["rush_yards"] / rb["carries"], 0.0)
    rb["ypr"] = np.where(rb["receptions"] > 0, rb["rec_yards"] / rb["receptions"], 0.0)
    rb["ypt"] = np.where(rb["targets"] > 0, rb["rec_yards"] / rb["targets"], 0.0)
    rb["yac_per_rec"] = np.where(rb["receptions"] > 0, rb["yac"] / rb["receptions"], 0.0)

    # Workload / context
    rb["scrimmage_yards"] = rb["rush_yards"] + rb["rec_yards"]
    rb["touches"] = rb["carries"] + rb["receptions"]
    rb["opportunities"] = rb["carries"] + rb["targets"]
    rb["yards_per_touch"] = np.where(rb["touches"] > 0, rb["scrimmage_yards"] / rb["touches"], 0.0)
    rb["td_total"] = rb["rush_td"] + rb["rec_td"]
    rb["fumbles_lost_total"] = rb["fumbles_lost_rush"] + rb["fumbles_lost_rec"]

    # Shares
    rb["rush_share"] = np.where(rb["team_carries"] > 0, rb["carries"] / rb["team_carries"], 0.0)
    rb["target_share"] = np.where(rb["team_targets"] > 0, rb["targets"] / rb["team_targets"], 0.0)

    # Provisional games = weeks with a carry OR target (replaced by PPR games after merge)
    inv_rows = pd.concat([
        rush_rows[["season","week","rusher_player_id"]].rename(columns={"rusher_player_id":"pid"}),
        tgt_rows[["season","week","receiver_player_id"]].rename(columns={"receiver_player_id":"pid"})
    ], ignore_index=True).dropna(subset=["pid"])

    inv_weeks = (
        inv_rows.groupby(["season","pid"], as_index=False)["week"]
                .nunique()
                .rename(columns={"pid":"player_id","week":"games"})
    )

    rb = rb.merge(inv_weeks, on=["season","player_id"], how="left")
    rb["games"] = rb["games"].fillna(0).astype(float)

    # Final keys for merge with PPR leaders
    rb = rb.rename(columns={"player_id":"fantasy_player_id","player_name":"fantasy_player_name"})

    # Column order
    cols_order = [
        "season","fantasy_player_id","fantasy_player_name","team",
        # Volume
        "carries","rush_yards","rush_td","gl_carries","i10_carries",
        "targets","receptions","rec_yards","rec_td",
        # Efficiency
        "ypc","ypr","ypt","yac","air_yards","yac_per_rec",
        # EPA summaries
        "rush_epa_sum","rec_epa_sum",
        # Team context
        "team_carries","team_targets","rush_share","target_share",
        # Workload
        "scrimmage_yards","touches","opportunities","yards_per_touch","td_total",
        # Ball security
        "fumbles_lost_rush","fumbles_lost_rec","fumbles_lost_total",
        # Games
        "games"
    ]
    rb = rb[[c for c in cols_order if c in rb.columns]]

    return rb


def build_rb_features(seasons):
    frames = []
    for yr in seasons:
        print(f"Engineering RB features for {yr} ...")
        frames.append(engineer_rb_features_for_season(yr))
    out = pd.concat(frames, ignore_index=True)

    out = out[
        (out["carries"] >= 100) |
        ((out["carries"] >= 60) & (out["targets"] >= 30)) |
        (out["opportunities"] >= 120)
    ].copy()
    return out


def merge_rb_with_targets(rb_feats: pd.DataFrame, players_leaders_multi: pd.DataFrame) -> pd.DataFrame:
    needed = {"season","fantasy_player_id","fp","games"}
    missing = needed - set(players_leaders_multi.columns)
    if missing:
        raise ValueError(f"players_leaders_multi missing columns: {missing}")
    assert not players_leaders_multi.duplicated(["season","fantasy_player_id"]).any(), \
        "players_leaders_multi must be unique per season+player (dedupe earlier)."

    # ---- Bring in positions from rosters and keep only RBs ----
    seasons_key = tuple(sorted(_to_py_int_list(rb_feats["season"].unique())))
    positions_tbl = load_positions_for(seasons_key)

    rb_pos = rb_feats.merge(
        positions_tbl, on=["season","fantasy_player_id"], how="left", validate="many_to_one"
    )
    pre_n = len(rb_pos)
    rb_pos = rb_pos[rb_pos["position"].astype(str).str.upper().eq("RB")].copy()
    post_n = len(rb_pos)
    print(f"RB position filter: kept {post_n}/{pre_n} rows where position == 'RB'")

    # Merge FP and PPR-defined games
    rb = rb_pos.drop(columns=["games"], errors="ignore").merge(
        players_leaders_multi[["season","fantasy_player_id","fp","games"]]
            .rename(columns={"games":"games_ppr"}),
        on=["season","fantasy_player_id"],
        how="left",
        validate="many_to_one"
    )
    rb["fp"] = rb["fp"].fillna(0.0)
    rb = rb.rename(columns={"games_ppr":"games"})

    rb = rb[rb["games"] >= 4].copy()
    return rb

rb_feats = build_rb_features(ALL_SEASONS)
rb_df = merge_rb_with_targets(rb_feats, players_leaders_multi)

# ================================
# TE dataframe construction (final, PPR-aligned games + position filter)
# Reuses from PPR script:
#   - ALL_SEASONS
#   - fantasy_week_max(season)
#   - base_filter(df, week_min, week_max)
#   - safe_col(df, name, default=0)
#   - normalize_team(team, season)
#   - players_leaders_multi with unique ['season','fantasy_player_id'] and cols ['fp','games','fantasy_player_name']
#   - import nflreadpy as nfl
# ================================

from datetime import datetime
import numpy as np
import pandas as pd

def _ensure_te_cols(df: pd.DataFrame) -> pd.DataFrame:
    """Ensure columns exist/cast for stable TE aggregations across vintages."""
    # Flags / ints
    must_int = ["pass_attempt", "rush_attempt", "complete_pass", "pass_touchdown", "rush_touchdown", "fumble_lost"]
    # Totals / floats
    must_float = ["receiving_yards", "rushing_yards", "air_yards", "yards_after_catch", "epa"]

    if "pass_attempt" not in df.columns and "pass" in df.columns:
        df["pass_attempt"] = df["pass"]
    if "rush_attempt" not in df.columns and "rush" in df.columns:
        df["rush_attempt"] = df["rush"]

    for c in must_int:
        if c not in df.columns:
            df[c] = 0
        df[c] = df[c].fillna(0).astype("int64")

    for c in must_float:
        if c not in df.columns:
            df[c] = 0.0
        df[c] = df[c].astype("float64").fillna(0.0)

    return df


def engineer_te_features_for_season(season: int) -> pd.DataFrame:
    pbp_pl = nfl.load_pbp([season])
    pbp = pbp_pl.to_pandas()
    wk_max = fantasy_week_max(season)
    pbp = base_filter(pbp, week_min=1, week_max=wk_max)

    pbp = _ensure_te_cols(pbp)

    # ---------- Receiving (targets & receptions credited to receiver role) ----------
    is_pass = (pbp["pass_attempt"] == 1)
    has_recv = pbp["receiver_player_id"].notna()

    tgt_rows = pbp[is_pass & has_recv].copy()
    rec_rows = pbp[(pbp["complete_pass"] == 1) & has_recv].copy()

    # Fumbles lost credited to receiver
    rec_rows["fl_rec"] = (
        (rec_rows["fumble_lost"] == 1) &
        (rec_rows["receiver_player_id"] == rec_rows.get("fumbled_1_player_id"))
    ).astype(int)

    tgt_g = (
        tgt_rows.groupby(["season","receiver_player_id","receiver_player_name"], as_index=False)
        .agg(targets=("play_id","count"))
        .rename(columns={"receiver_player_id":"player_id","receiver_player_name":"player_name"})
    )

    rec_g = (
        rec_rows.groupby(["season","receiver_player_id","receiver_player_name"], as_index=False)
        .agg(
            receptions=("complete_pass","sum"),
            rec_yards=("receiving_yards","sum"),
            rec_td=("pass_touchdown","sum"),
            air_yards=("air_yards","sum"),
            yac=("yards_after_catch","sum"),
            rec_epa_sum=("epa","sum"),
            fumbles_lost_rec=("fl_rec","sum")
        )
        .rename(columns={"receiver_player_id":"player_id","receiver_player_name":"player_name"})
    )

    # ---------- TE rushing ----------
    rush_rows = pbp[(pbp["rush_attempt"] == 1) & pbp["rusher_player_id"].notna()].copy()
    # Fumbles lost credited to the rusher
    rush_rows["fl_rush"] = (
        (rush_rows["fumble_lost"] == 1) &
        (rush_rows["rusher_player_id"] == rush_rows.get("fumbled_1_player_id"))
    ).astype(int)

    rush_g = (
        rush_rows.groupby(["season","rusher_player_id","rusher_player_name"], as_index=False)
        .agg(
            te_rush_att=("rush_attempt","sum"),
            te_rush_yards=("rushing_yards","sum"),
            te_rush_td=("rush_touchdown","sum"),
            fumbles_lost_rush=("fl_rush","sum"),
        )
        .rename(columns={"rusher_player_id":"player_id","rusher_player_name":"player_name"})
    )

    # ---------- Combine receiving + rushing ----------
    te = (
        tgt_g.merge(rec_g, on=["season","player_id","player_name"], how="outer")
             .merge(rush_g, on=["season","player_id","player_name"], how="left")
             .fillna({
                 "targets":0, "receptions":0, "rec_yards":0.0, "rec_td":0,
                 "air_yards":0.0, "yac":0.0, "rec_epa_sum":0.0, "fumbles_lost_rec":0,
                 "te_rush_att":0, "te_rush_yards":0.0, "te_rush_td":0, "fumbles_lost_rush":0,
             })
    )

    # ---------- Modal team per player-season ----------
    team_source = pd.concat([
        tgt_rows[["season","receiver_player_id","posteam"]].rename(columns={"receiver_player_id":"pid","posteam":"team"}),
        rush_rows[["season","rusher_player_id","posteam"]].rename(columns={"rusher_player_id":"pid","posteam":"team"}),
    ], ignore_index=True)

    team_mode = (
        team_source.dropna(subset=["pid","team"])
                   .groupby(["season","pid"])["team"]
                   .agg(lambda s: s.dropna().astype(str).mode().iloc[0] if len(s.dropna()) > 0 else np.nan)
                   .reset_index()
                   .rename(columns={"pid":"player_id","team":"team"})
    )

    te = te.merge(team_mode, on=["season","player_id"], how="left")
    if "team" in te.columns:
        te["team"] = [normalize_team(str(t), int(s)) if pd.notna(t) else np.nan
                      for t, s in zip(te["team"], te["season"])]

    # ---------- Team targets & target share ----------
    team_tgts = (
        tgt_rows.groupby(["season","posteam"], as_index=False)
                .agg(team_targets=("play_id","count"))
                .rename(columns={"posteam":"team_raw"})
    )
    te = te.merge(team_tgts, left_on=["season","team"], right_on=["season","team_raw"], how="left")
    te = te.drop(columns=["team_raw"], errors="ignore")
    te["team_targets"] = te["team_targets"].fillna(0).astype(float)

    # ---------- Feature engineering ----------
    te["adot"] = np.where(te["targets"] > 0, te["air_yards"] / te["targets"], 0.0)
    te["ypr"] = np.where(te["receptions"] > 0, te["rec_yards"] / te["receptions"], 0.0)
    te["ypt"] = np.where(te["targets"] > 0, te["rec_yards"] / te["targets"], 0.0)
    te["yac_per_rec"] = np.where(te["receptions"] > 0, te["yac"] / te["receptions"], 0.0)
    te["target_share"] = np.where(te["team_targets"] > 0, te["targets"] / te["team_targets"], 0.0)

    # Provisional games = #weeks with target or TE rush
    inv_rows = pd.concat([
        tgt_rows[["season","week","receiver_player_id"]].rename(columns={"receiver_player_id":"pid"}),
        rush_rows[["season","week","rusher_player_id"]].rename(columns={"rusher_player_id":"pid"})
    ], ignore_index=True).dropna(subset=["pid"])
    inv_weeks = (
        inv_rows.groupby(["season","pid"], as_index=False)["week"]
                .nunique()
                .rename(columns={"pid":"player_id","week":"games"})
    )
    te = te.merge(inv_weeks, on=["season","player_id"], how="left")
    te["games"] = te["games"].fillna(0).astype(float)

    te = te.rename(columns={"player_id":"fantasy_player_id","player_name":"fantasy_player_name"})

    cols_order = [
        "season","fantasy_player_id","fantasy_player_name","team",
        # Receiving
        "targets","receptions","rec_yards","rec_td","air_yards","yac",
        "adot","ypr","ypt","yac_per_rec",
        # Team context
        "team_targets","target_share",
        # Rushing
        "te_rush_att","te_rush_yards","te_rush_td",
        # EPA
        "rec_epa_sum",
        # Games
        "games"
    ]
    te = te[[c for c in cols_order if c in te.columns]]
    return te


# ---- Positions helper ----
def load_positions_for(seasons) -> pd.DataFrame:
    """Load (season, fantasy_player_id, position) from nflreadpy rosters; cast seasons to ints safely."""
    seasons = pd.Series(seasons).dropna().astype(int)
    upper = max(datetime.now().year, 2025)
    seasons = seasons[(seasons >= 1920) & (seasons <= upper)].unique().tolist()
    if not seasons:
        raise ValueError("No valid seasons to load positions for.")

    rosters_pl = nfl.load_rosters(seasons)
    rosters = rosters_pl.to_pandas()

    id_col = "gsis_id" if "gsis_id" in rosters.columns else ("player_id" if "player_id" in rosters.columns else None)
    if id_col is None:
        raise KeyError("No GSIS id column found in rosters (expected 'gsis_id' or 'player_id').")

    pos_col = "position" if "position" in rosters.columns else ("position_group" if "position_group" in rosters.columns else None)
    if pos_col is None:
        raise KeyError("No position column found in rosters (expected 'position' or 'position_group').")

    keep = rosters[["season", id_col, pos_col]].copy()
    keep = keep.rename(columns={id_col: "fantasy_player_id", pos_col: "position"})
    keep = (keep.sort_values(["season","fantasy_player_id"])
                .groupby(["season","fantasy_player_id"], as_index=False)
                .agg(position=("position","first")))
    return keep


def build_te_features(seasons):
    frames = []
    for yr in seasons:
        print(f"Engineering TE features for {yr} ...")
        frames.append(engineer_te_features_for_season(yr))
    out = pd.concat(frames, ignore_index=True)

    out = out[
        (out["targets"] >= 40) |
        ((out["targets"] >= 25) & (out["games"] >= 4))
    ].copy()
    return out


def merge_te_with_targets(te_feats: pd.DataFrame, players_leaders_multi: pd.DataFrame) -> pd.DataFrame:
    needed = {"season","fantasy_player_id","fp","games"}
    missing = needed - set(players_leaders_multi.columns)
    if missing:
        raise ValueError(f"players_leaders_multi missing columns: {missing}")
    assert not players_leaders_multi.duplicated(["season","fantasy_player_id"]).any(), \
        "players_leaders_multi must be unique per season+player (dedupe earlier)."

    # Attach positions and filter to tight ends
    pos_tbl = load_positions_for(te_feats["season"].unique())
    before = len(te_feats)
    te_feats = te_feats.merge(pos_tbl, on=["season","fantasy_player_id"], how="left", validate="many_to_one")
    te_feats = te_feats[te_feats["position"] == "TE"].copy()
    after = len(te_feats)
    print(f"TE position filter: kept {after}/{before} rows where position == 'TE'")

    # Merge FP and PPR-defined games
    te = te_feats.drop(columns=["games"], errors="ignore").merge(
        players_leaders_multi[["season","fantasy_player_id","fp","games"]]
            .rename(columns={"games":"games_ppr"}),
        on=["season","fantasy_player_id"],
        how="left",
        validate="many_to_one"
    )
    te["fp"] = te["fp"].fillna(0.0)
    te = te.rename(columns={"games_ppr":"games"})

    te = te[te["games"] >= 4].copy()
    return te

te_feats = build_te_features(ALL_SEASONS)
te_df = merge_te_with_targets(te_feats, players_leaders_multi)

# ================================
# K dataframe construction (final, PPR-aligned games)
# Reuses from our PPR script:
#   - ALL_SEASONS
#   - fantasy_week_max(season)
#   - base_filter(df, week_min, week_max)
#   - normalize_team(team, season)
#   - k_season_multi from kicker_scoring()  (season totals)
#   - import nflreadpy as nfl
#   - import pandas as pd, import numpy as np
# ================================

# ---------- helpers ----------

def _ensure_k_cols(df: pd.DataFrame) -> pd.DataFrame:
    """Ensure columns exist/cast for stable K aggregations across vintages."""
    # Flags / ints
    must_int = ["field_goal_attempt", "extra_point_attempt"]
    # Results to lowercase strings if present
    str_cols = ["field_goal_result", "extra_point_result"]
    # Distances
    float_cols = ["kick_distance"]

    # Backfill legacy flags if needed
    if "field_goal_attempt" not in df.columns and "field_goal_result" in df.columns:
        df["field_goal_attempt"] = (df["field_goal_result"].notna()).astype("int64")
    if "extra_point_attempt" not in df.columns and "extra_point_result" in df.columns:
        df["extra_point_attempt"] = (df["extra_point_result"].notna()).astype("int64")

    for c in must_int:
        if c not in df.columns:
            df[c] = 0
        df[c] = df[c].fillna(0).astype("int64")

    for c in str_cols:
        if c in df.columns:
            df[c] = df[c].astype(str).str.lower()
        else:
            df[c] = ""

    for c in float_cols:
        if c not in df.columns:
            df[c] = 0.0
        df[c] = df[c].astype("float64").fillna(0.0)

    return df


def _bucket_fg_points(is_made: pd.Series, dist: pd.Series) -> pd.Series:
    """ESPN-like FG made buckets: 0-39 = 3, 40-49 = 4, 50-59 = 5, 60+ = 6."""
    return np.where(is_made & (dist >= 60), 6,
           np.where(is_made & (dist >= 50), 5,
           np.where(is_made & (dist >= 40), 4,
           np.where(is_made, 3, 0))))


def _kicker_name_cols(df: pd.DataFrame) -> tuple[str, str]:
    """Return the best-available (id_col, name_col) for the kickers table."""
    id_col = "kicker_player_id" if "kicker_player_id" in df.columns else "fantasy_player_id"
    name_col = "kicker_player_name" if "kicker_player_name" in df.columns else "fantasy_player_name"
    return id_col, name_col


# ---------- feature engineering per season ----------

def engineer_k_features_for_season(season: int, miss_pat_minus_one: bool = False) -> pd.DataFrame:
    pbp_pl = nfl.load_pbp([season])
    pbp = pbp_pl.to_pandas()
    wk_max = fantasy_week_max(season)
    pbp = base_filter(pbp, week_min=1, week_max=wk_max)

    pbp = _ensure_k_cols(pbp)

    # Keep only rows with any kicking relevance
    kick = pbp[(pbp["field_goal_attempt"] == 1) | (pbp["extra_point_attempt"] == 1)].copy()

    # Identify id/name columns present
    id_col, name_col = _kicker_name_cols(kick)

    # Normalize posteam historically for outputs
    kick["team"] = [normalize_team(str(t), int(s)) if pd.notna(t) else np.nan
                    for t, s in zip(kick.get("posteam", ""), kick["season"])]

    # Results flags
    dist = kick["kick_distance"].astype(float)
    fg_attempt = (kick["field_goal_attempt"] == 1)
    fg_made = fg_attempt & (kick["field_goal_result"].isin(["made", "good"]))  # older feeds sometimes "good"
    fg_missed = fg_attempt & (kick["field_goal_result"].isin(["missed", "blocked"]))

    # Distance buckets
    fg_0_39_made = (fg_made & (dist < 40)).astype(int)
    fg_40_49_made = (fg_made & (40 <= dist) & (dist < 50)).astype(int)
    fg_50_59_made = (fg_made & (50 <= dist) & (dist < 60)).astype(int)
    fg_60p_made   = (fg_made & (dist >= 60)).astype(int)

    # PATs
    pat_attempt = (kick["extra_point_attempt"] == 1)
    pat_good = pat_attempt & (kick["extra_point_result"].isin(["good", "made"]))
    pat_miss = pat_attempt & (kick["extra_point_result"].isin(["failed", "missed", "blocked"]))

    # Per-play fantasy points
    k_fp_per_play = (
        _bucket_fg_points(fg_made, dist)
        + np.where(fg_missed, -1, 0)
        + np.where(pat_good, 1, 0)
        + np.where(miss_pat_minus_one & pat_miss, -1, 0)
    )

    kick["fg_0_39_made"] = fg_0_39_made
    kick["fg_40_49_made"] = fg_40_49_made
    kick["fg_50_59_made"] = fg_50_59_made
    kick["fg_60p_made"] = fg_60p_made
    kick["fg_miss"] = fg_missed.astype(int)
    kick["pat_miss"] = pat_miss.astype(int)
    kick["k_fp_calc"] = k_fp_per_play

    # Aggregate per kicker-season
    g = (
        kick.groupby(["season", id_col, name_col, "team"], as_index=False)
            .agg(
                fg_att=("field_goal_attempt", "sum"),
                fg_made=("field_goal_result", lambda s: int((s.isin(["made", "good"])).sum())),
                fg_0_39_made=("fg_0_39_made", "sum"),
                fg_40_49_made=("fg_40_49_made", "sum"),
                fg_50_59_made=("fg_50_59_made", "sum"),
                fg_60p_made=("fg_60p_made", "sum"),
                fg_miss=("fg_miss", "sum"),
                pat_att=("extra_point_attempt", "sum"),
                pat_made=("extra_point_result", lambda s: int((s.isin(["good", "made"])).sum())),
                pat_miss=("pat_miss", "sum"),
                k_fp_calc=("k_fp_calc", "sum"),
                games=("game_id", "nunique"),
            )
    )

    # Align keys to canonical names
    g = g.rename(columns={
        id_col: "fantasy_player_id",
        name_col: "fantasy_player_name"
    })

    return g


def build_k_features(seasons, miss_pat_minus_one: bool = False) -> pd.DataFrame:
    frames = []
    for yr in seasons:
        print(f"Engineering K features for {yr} ...")
        frames.append(engineer_k_features_for_season(yr, miss_pat_minus_one=miss_pat_minus_one))
    out = pd.concat(frames, ignore_index=True)

    # ---------- attach positions from nflreadpy rosters & filter to 'K' ----------
    from nflreadpy import load_rosters
    rosters_pl = load_rosters(list(set(out["season"].astype(int).tolist())))
    rosters = rosters_pl.to_pandas()

    # Normalize roster columns
    if "gsis_id" not in rosters.columns:
        rosters["gsis_id"] = rosters.get("player_id", None)
    if "position" not in rosters.columns:
        rosters["position"] = ""
    if "season" not in rosters.columns:
        rosters["season"] = rosters.get("year", out["season"].min())

    pos_tbl = (
        rosters[["season", "gsis_id", "position"]]
        .dropna(subset=["gsis_id"])
        .rename(columns={"gsis_id": "fantasy_player_id"})
    )

    # Collapse to one row per (season, player): prefer the most frequent non-null position.
    pos_tbl = (
        pos_tbl.groupby(["season", "fantasy_player_id"])["position"]
               .agg(lambda s: s.dropna().astype(str).mode().iloc[0] if len(s.dropna()) else "")
               .reset_index()
    )

    out = out.merge(pos_tbl, on=["season", "fantasy_player_id"], how="left", validate="many_to_one")
    before = len(out)
    out = out[out["position"] == "K"].copy()
    print(f"K position filter: kept {len(out)}/{before} rows where position == 'K'")

    # at least 4 credited games 
    out = out[out["games"] >= 4].copy()

    return out


# ---------- merge with season totals from kicker_scoring() ----------

def merge_k_with_totals(k_feats: pd.DataFrame, k_season_multi: pd.DataFrame) -> pd.DataFrame:
    """
    Merge engineered kicker features with season totals from kicker_scoring().
    - Merge keys: (season, fantasy_player_id) ONLY.
    - Dedup totals first (keeps the highest total_fp, then most games).
    - Coalesce a safe fantasy_player_name from available columns.
    """

    def _dedupe_k_totals(df: pd.DataFrame) -> pd.DataFrame:
        tot = df.rename(columns={
            "kicker_player_id": "fantasy_player_id",
            "kicker_player_name": "fantasy_player_name",
            "posteam": "team_ppr",
            "total_fp": "fp_ppr"
        }).copy()

        tot = (tot.sort_values(["season","fantasy_player_id","fp_ppr","games"],
                               ascending=[True, True, False, False])
                   .drop_duplicates(["season","fantasy_player_id"], keep="first"))
        return tot

    k_totals = _dedupe_k_totals(k_season_multi)

    right_cols = [
        "season", "fantasy_player_id", "fantasy_player_name", "team_ppr",
        "fp_ppr", "games", "fg_made", "fg_att", "pat_made", "pat_att", "ppg"
    ]
    right_cols = [c for c in right_cols if c in k_totals.columns]
    k = k_feats.merge(
        k_totals[right_cols],
        on=["season", "fantasy_player_id"],
        how="left",
        validate="many_to_one"
    )

    def _series(df, col):
        return df[col] if col in df.columns else pd.Series(index=df.index, dtype="object")

    name_engineered = _series(k, "fantasy_player_name")
    name_ppr_y = _series(k, "fantasy_player_name_y")
    name_ppr = name_ppr_y if not name_ppr_y.isna().all() else _series(k, "fantasy_player_name")
    name_kicker = _series(k, "kicker_player_name")

    k["fantasy_player_name"] = name_engineered.fillna(name_ppr).fillna(name_kicker)

    # season fantasy points to PPR totals
    if "fp_ppr" in k.columns:
        k = k.rename(columns={"fp_ppr": "fp"})

    keep_order = [
        "season", "fantasy_player_id", "fantasy_player_name",
        "team", "team_ppr",
        # volume
        "fg_att", "fg_made", "pat_att", "pat_made",
        # distance buckets
        "fg_0_39_made", "fg_40_49_made", "fg_50_59_made", "fg_60p_made",
        "fg_miss", "pat_miss",
        # scoring
        "k_fp_calc", "fp", "ppg",
        # games & position
        "games", "position"
    ]
    keep_order = [c for c in keep_order if c in k.columns]
    k = k[keep_order].copy()

    if "position" in k.columns:
        k = k[k["position"] == "K"].copy()
    if "games" in k.columns:
        k = k[k["games"] >= 4].copy()

    return k

k_feats = build_k_features(ALL_SEASONS, miss_pat_minus_one=False)
k_df = merge_k_with_totals(k_feats, k_season_multi)

# ================================
# D/ST dataframe construction (final, PPR-aligned games)
# Reuses from PPR script:
#   - ALL_SEASONS
#   - fantasy_week_max(season)
#   - base_filter(df, week_min, week_max)
#   - normalize_team(team, season)
#   - dst_scoring() output: dst_season_multi (season totals)
#   - import nflreadpy as nfl
#   - import pandas as pd, import numpy as np
# ================================

# ---------- helpers ----------

def _ensure_dst_cols(df: pd.DataFrame) -> pd.DataFrame:
    """Make older/varying seasons safe for D/ST aggregation."""
    must_int = [
        "sack", "interception", "safety", "punt_blocked",
        "field_goal_attempt", 
        "return_touchdown"
    ]
    must_str = ["field_goal_result", "extra_point_result"]
    for c in must_int:
        if c not in df.columns: df[c] = 0
        df[c] = df[c].fillna(0).astype("int64")

    for c in must_str:
        if c in df.columns:
            df[c] = df[c].astype(str).str.lower()
        else:
            df[c] = ""

    if "return_touchdown" not in df.columns:
        df["return_touchdown"] = 0

    df["fg_block"] = (df["field_goal_result"] == "blocked").astype("int64")

    for c in ["total_home_score","total_away_score","home_team","away_team"]:
        if c not in df.columns:
            df[c] = np.nan

    for c in ["defteam","return_team"]:
        if c not in df.columns:
            df[c] = ""

    if "fumble_recovery_1_team" not in df.columns:
        df["fumble_recovery_1_team"] = np.nan
    if "fumble_recovery_2_team" not in df.columns:
        df["fumble_recovery_2_team"] = np.nan

    return df


def _pa_to_pts(pa: int) -> int:
    """Points-allowed → fantasy points (ESPN-ish; matches your dst_scoring)."""
    if pa == 0: return 10
    if pa <= 6: return 7
    if pa <= 13: return 4
    if pa <= 17: return 1
    if pa <= 27: return 0
    if pa <= 34: return -1
    if pa <= 45: return -4
    return -5


# ---------- feature engineering per season ----------

def engineer_dst_features_for_season(season: int) -> pd.DataFrame:
    pbp_pl = nfl.load_pbp([season])
    pbp = pbp_pl.to_pandas()
    wk_max = fantasy_week_max(season)
    pbp = base_filter(pbp, week_min=1, week_max=wk_max)

    pbp = _ensure_dst_cols(pbp)

    # Final scores → Points Allowed per team per game
    scores = (
        pbp.groupby(["season","game_id"], as_index=False)[
            ["total_home_score","total_away_score","home_team","away_team"]
        ]
        .agg({"total_home_score":"max","total_away_score":"max","home_team":"first","away_team":"first"})
        .rename(columns={"total_home_score":"home_pts_final","total_away_score":"away_pts_final"})
    )

    home_rows = scores[["season","game_id","home_team","away_pts_final"]].rename(
        columns={"home_team":"team","away_pts_final":"points_allowed"})
    away_rows = scores[["season","game_id","away_team","home_pts_final"]].rename(
        columns={"away_team":"team","home_pts_final":"points_allowed"})
    pa_tbl = pd.concat([home_rows, away_rows], ignore_index=True)

    pa_tbl["team"] = [normalize_team(str(t), int(season)) for t in pa_tbl["team"].astype(str)]

    pa_tbl["pa_pts"] = pa_tbl["points_allowed"].astype(int).map(_pa_to_pts)

    # Defensive tallies per game
    by_def = (
        pbp.groupby(["season","game_id","defteam"], as_index=False)
           .agg({"sack":"sum","interception":"sum","safety":"sum","punt_blocked":"sum"})
           .rename(columns={"defteam":"team"})
    )
    by_def["team"] = [normalize_team(str(t), int(season)) for t in by_def["team"].astype(str)]

    # Blocked FGs per game
    by_def_fgblk = (
        pbp.groupby(["season","game_id","defteam"], as_index=False)["fg_block"].sum()
           .rename(columns={"defteam":"team"})
    )
    by_def_fgblk["team"] = [normalize_team(str(t), int(season)) for t in by_def_fgblk["team"].astype(str)]

    # Fumble recoveries credited to recovering team
    fr_parts = []
    fr1 = pbp[pbp["fumble_recovery_1_team"].notna()][["season","game_id","fumble_recovery_1_team"]].copy()
    if len(fr1):
        fr1 = fr1.rename(columns={"fumble_recovery_1_team":"team"}); fr1["fr"] = 1
        fr_parts.append(fr1[["season","game_id","team","fr"]])
    fr2 = pbp[pbp["fumble_recovery_2_team"].notna()][["season","game_id","fumble_recovery_2_team"]].copy()
    if len(fr2):
        fr2 = fr2.rename(columns={"fumble_recovery_2_team":"team"}); fr2["fr"] = 1
        fr_parts.append(fr2[["season","game_id","team","fr"]])

    if fr_parts:
        by_fr = (pd.concat(fr_parts, ignore_index=True)
                 .groupby(["season","game_id","team"], as_index=False)["fr"].sum())
        by_fr["team"] = [normalize_team(str(t), int(season)) for t in by_fr["team"].astype(str)]
    else:
        by_fr = pd.DataFrame(columns=["season","game_id","team","fr"])

    # Return TDs credited to return team
    ret = pbp[(pbp["return_touchdown"] == 1)].copy()
    by_ret_td = (ret.groupby(["season","game_id","return_team"], as_index=False).size()
                   .rename(columns={"return_team":"team","size":"ret_td"}))
    if len(by_ret_td):
        by_ret_td["team"] = [normalize_team(str(t), int(season)) for t in by_ret_td["team"].astype(str)]

    # Merge components to game-level table
    cols_merge = ["season","game_id","team"]
    dst_g = pa_tbl[cols_merge + ["points_allowed","pa_pts"]].copy()

    for part in [
        by_def[cols_merge + ["sack","interception","safety","punt_blocked"]],
        by_def_fgblk[cols_merge + ["fg_block"]],
        by_fr[cols_merge + ["fr"]],
        by_ret_td[cols_merge + ["ret_td"]] if len(by_ret_td) else pd.DataFrame(columns=cols_merge+["ret_td"])
    ]:
        if len(part):
            dst_g = dst_g.merge(part, on=cols_merge, how="left")

    for c in ["sack","interception","safety","punt_blocked","fg_block","fr","ret_td"]:
        if c not in dst_g.columns: dst_g[c] = 0
        dst_g[c] = dst_g[c].fillna(0).astype(int)
    dst_g["pa_pts"] = dst_g["pa_pts"].fillna(0).astype(int)

    # Fantasy points per game
    dst_g["dst_fp_calc"] = (
        dst_g["sack"]*1 + dst_g["interception"]*2 + dst_g["fr"]*2 + dst_g["safety"]*2
        + dst_g["ret_td"]*6 + (dst_g["punt_blocked"] + dst_g["fg_block"])*2 + dst_g["pa_pts"]
    )

    # Aggregate per season
    agg = (
        dst_g.groupby(["season","team"], as_index=False)
             .agg(
                 games=("game_id","nunique"),
                 points_allowed_sum=("points_allowed","sum"),
                 sacks=("sack","sum"),
                 ints=("interception","sum"),
                 fr=("fr","sum"),
                 safeties=("safety","sum"),
                 blk_punts=("punt_blocked","sum"),
                 blk_fgs=("fg_block","sum"),
                 ret_td=("ret_td","sum"),
                 pa_pts_sum=("pa_pts","sum"),
                 dst_fp_calc=("dst_fp_calc","sum"),
             )
    )
    agg["ppg_calc"] = agg["dst_fp_calc"] / agg["games"].replace(0, np.nan)

    agg["sacks_pg"] = agg["sacks"] / agg["games"].replace(0, np.nan)
    agg["takeaways"] = agg["ints"] + agg["fr"]
    agg["takeaways_pg"] = agg["takeaways"] / agg["games"].replace(0, np.nan)

    return agg, dst_g


def build_dst_features(seasons) -> pd.DataFrame:
    frames = []
    for yr in seasons:
        print(f"Engineering D/ST features for {yr} ...")
        season_agg, _ = engineer_dst_features_for_season(yr)
        frames.append(season_agg)
    out = pd.concat(frames, ignore_index=True)

    out = out[out["games"] >= 4].copy()

    # column order
    cols = [
        "season","team","games",
        "points_allowed_sum","pa_pts_sum",
        "sacks","ints","fr","safeties","blk_punts","blk_fgs","ret_td",
        "takeaways","dst_fp_calc","ppg_calc",
        "sacks_pg","takeaways_pg"
    ]
    out = out[[c for c in cols if c in out.columns]]
    return out


# ---------- merge with season totals from dst_scoring() ----------

def merge_dst_with_totals(dst_feats: pd.DataFrame, dst_season_multi: pd.DataFrame) -> pd.DataFrame:
    """
    Merge engineered D/ST features with season totals from dst_scoring().
    Keys: (season, team). Assumes dst_season_multi has columns ['season','team','dst_fp'].
    """
    right = dst_season_multi.rename(columns={"dst_fp":"fp"}).copy()
    right = (right.sort_values(["season","team","fp"], ascending=[True, True, False])
                  .drop_duplicates(["season","team"], keep="first"))

    dst = dst_feats.merge(right[["season","team","fp"]], on=["season","team"], how="left", validate="one_to_one")

    if "dst_fp_calc" in dst.columns and "fp" in dst.columns:
        dst["fp_delta"] = (dst["dst_fp_calc"] - dst["fp"]).round(3)
    return dst

dst_feats = build_dst_features(ALL_SEASONS)
dst_df = merge_dst_with_totals(dst_feats, dst_season_multi)

# ================================
# Save all position dataframes as CSV
# ================================

import os

# Create an output directory
output_dir = "fantasy_features_csv"
os.makedirs(output_dir, exist_ok=True)

# Save each dataframe to CSV
qb_df.to_csv(os.path.join(output_dir, "QB_features.csv"), index=False)
rb_df.to_csv(os.path.join(output_dir, "RB_features.csv"), index=False)
wr_df.to_csv(os.path.join(output_dir, "WR_features.csv"), index=False)
te_df.to_csv(os.path.join(output_dir, "TE_features.csv"), index=False)
k_df.to_csv(os.path.join(output_dir, "K_features.csv"), index=False)
dst_df.to_csv(os.path.join(output_dir, "DST_features.csv"), index=False)

print("✅ All CSV files saved successfully in:", os.path.abspath(output_dir))