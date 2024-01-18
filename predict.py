import pandas as pd
import numpy as np
from statsmodels import api as sm
from scipy.stats import poisson


def team_goals_parse(s):
    if s.Result is np.NaN:
        return s
    parts = s.Result.split(" - ")
    s["HomeGoals"] = int(parts[0])
    s["AwayGoals"] = int(parts[1])
    return s


def simulate_match(series, model, max_goals):
    home = series["HomeTeam"]
    away = series["AwayTeam"]

    home_goals_avg = model.predict(pd.DataFrame(data={"scoring": home, "conceding": away, "home_status": 1},
                                                index=[0]))
    away_goals_avg = model.predict(pd.DataFrame(data={"scoring": away, "conceding": home, "home_status": 0},
                                                index=[0]))
    predictions = [[poisson.pmf(i, team_avg) for i in range(0, max_goals + 1)]
                   for team_avg in [home_goals_avg, away_goals_avg]]
    simulation_results = np.outer(np.array(predictions[0]), np.array(predictions[1]))
    unraveled = np.unravel_index(simulation_results.argmax(), simulation_results.shape)
    series["HomeGoals"] = unraveled[0]
    series["AwayGoals"] = unraveled[1]
    series["Result"] = f"{unraveled[0]} - {unraveled[1]}"
    return series


def round_map(s):
    if s == "Wild Card Round":
        return 19
    elif s == "Divisional Round":
        return 20
    elif s == "Conference Champs":
        return 21
    elif s == "Super Bowl":
        return 22
    else:
        return s


if __name__ == '__main__':
    df = pd.read_csv("https://fixturedownload.com/download/nfl-2023-EasternStandardTime.csv")
    df = df.rename(columns={"Home Team": "HomeTeam", "Away Team": "AwayTeam"})
    df = df.apply(team_goals_parse, axis=1)
    df["Round Number"] = df["Round Number"].apply(round_map)
    df["Round Number"] = df["Round Number"].astype(int)

    df_played = df.loc[df["Result"].notnull()]
    df_futures = df.loc[df["Result"].isnull()]
    df_next_matches = df_futures[df_futures["Round Number"] == df_futures["Round Number"].to_numpy().min()]

    goal_data = pd.concat([df_played[["HomeTeam", "AwayTeam", "HomeGoals"]].assign(home_status=1).rename(
        columns={"HomeTeam": "scoring", "AwayTeam": "conceding", "HomeGoals": "goals"}),
        df_played[["AwayTeam", "HomeTeam", "AwayGoals"]].assign(home_status=0).rename(
            columns={"AwayTeam": "scoring", "HomeTeam": "conceding", "AwayGoals": "goals"})])

    mp = sm.GLM.from_formula(formula="goals ~ home_status + scoring + conceding",
                             data=goal_data,
                             family=sm.families.Poisson())

    poisson_model = mp.fit()

    df_next_matches = df_next_matches.apply(simulate_match, model=poisson_model, max_goals=75, axis=1)
    df_next_matches = df_next_matches[["HomeTeam", "AwayTeam", "Result"]]
    df_next_matches["X"] = df_next_matches["HomeTeam"].astype(str) + " (" + \
                           df_next_matches["Result"].astype(str) + ") " + \
                           df_next_matches["AwayTeam"].astype(str)
    print(str(df_next_matches["X"].values))
