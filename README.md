# CSE-6242-Group-Project
Git Hub for CSE 6242 Group Project

Variables

Added by our fantasy-scoring code

fp — Total player PPR fantasy points for a play (passing + rushing + receiving + 2-pt − INT − fumbles lost), later aggregated by player/week/season.

ppg — Points per game for players or kickers (total points ÷ games with credited points).

dst_fp — Defense/Special Teams fantasy points for a team in a given game (sacks, INTs, FRs, safeties, blocked kicks, return TDs, + points-allowed tier).

k_fp — Kicker fantasy points for a play (FG made by distance, FG missed = −1 incl. blocked, PAT made = +1; missed PAT penalty off by default).

pa_pts — Points-allowed tier bonus/penalty used in D/ST scoring, mapped from opponent final score.

points_allowed — Opponent final score used to derive pa_pts.

fg_block — Count of blocked field goals (derived from field_goal_result == "blocked"), credited to the defense in D/ST.

fr — Fumble recoveries credited to the recovering team (from fumble_recovery_1_team / _2_team) for D/ST.

ret_td — Return touchdowns credited to the return team (punt/kick/INT/FR returns) for D/ST.

games — Number of unique weeks/games a player/kicker was credited with >0 points (used for PPG).



Original nflverse / nflfastR PBP fields (grouped)


Core identifiers

play_id — Play identifier (unique with game_id and drive). 

game_id — Ten-char nflverse game ID. 

season, week, season_type — Season year, week, and whether REG/POST. 


Teams & possession

home_team, away_team — Home/away team abbreviations. 

posteam, defteam — Offense (in possession) and defense on the play. 

possession_team — Normalized possession team (post-processing). 


Clock & drive context

qtr, game_half, time, game_seconds_remaining, quarter_seconds_remaining, half_seconds_remaining — Timing indicators. 

drive, fixed_drive, drive_play_count, drive_start_yard_line, drive_end_yard_line, drive_time_of_possession, drive_result — Drive metadata. 


Field position & yards to go

yardline_100, side_of_field, ydstogo, goal_to_go — Ball position and distance to first down/goal. 


Play description & type

desc — Human-readable description string. 

play_type / play_type_nfl / st_play_type / special_teams_play — Play classification (pass, run, punt, kickoff, etc.). 

aborted_play, play_deleted — Flags for aborted or removed/invalid plays (we exclude these). 


Pass / run specifics

pass_attempt, rush_attempt, qb_dropback, complete_pass, incomplete_pass, sack, qb_scramble — Core outcomes. 

pass_length, pass_location, air_yards, yards_after_catch — Passing geometry. 

run_location, run_gap — Run direction/gap. 

yards_gained, passing_yards, rushing_yards, receiving_yards — Yardage accounting. 


Scoring events & conversions

touchdown, pass_touchdown, rush_touchdown, return_touchdown — TD indicators by type. 

extra_point_attempt, extra_point_result — PAT try and outcome. 

two_point_attempt, two_point_conv_result — 2-pt try and result (we use success only). 

field_goal_attempt, field_goal_result, kick_distance — FG try, outcome, distance (used for K scoring). 


Turnovers, penalties, misc.

interception — Interception thrown. 

fumble, fumble_lost, fumbled_1_player_id/team, fumble_recovery_1/2 fields* — Fumble and recovery details. 

penalty, penalty_team, penalty_player_id, penalty_type, penalty_yards — Penalty info (we exclude negated plays via play_type != "no_play"). 

safety — Safety indicator. 


Special teams & blocks

punt_attempt, punt_blocked, kickoff_attempt — ST attempts and blocked punts. 
NFLReadr

punter_player_, kicker_player_ , return_team, return_yards — Specialists and return team info. 
NFLReadr

Players on the play

passer_player_id/_name, rusher_player_id/_name, receiver_player_id/_name — Principal players credited. 

fantasy_player_id/_name — The player tied to fantasy attribution on that row (we group by this ID). 

players_on_play, offense_players, defense_players — Lists/counts of involved players. 


Win/score/EP (expected points) metrics

total_home_score, total_away_score, home_score, away_score, score_differential — Score tracking. 

ep, epa, wpa, wp, home_wp, away_wp (+ their air/YAC/comp/raw variants) — Expected/win probability analytics. 


Situational & model features

xpass, pass_oe — Model-based pass probability and pass rate over expected. 

xyac_ (mean, median, success, fd)* — Expected YAC family features. 


Drive/game meta & environment

roof, surface, temp, wind, weather, stadium, game_stadium — Venue and conditions. 

spread_line, total_line, vegas_wp, vegas_wpa — Betting/market features. 


Coverage & tracking (when available)

ngs_air_yards, time_to_throw, was_pressure, route — NGS-derived fields on some seasons. 

defense_man_zone_type, defense_coverage_type, number_of_pass_rushers, defenders_in_box — Defensive look/pressure descriptors.
