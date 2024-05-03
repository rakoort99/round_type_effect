import pandas as pd
import trueskillthroughtime as ttt
from datetime import datetime
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import argparse
sns.set_theme()
rounds_df = 'annotated_results_df_final.pkl'
teams_df = 'teams_scores_final.xlsx'
judge_df = 'judge_scores_final.xlsx'

parser = argparse.ArgumentParser("process_model")
parser.add_argument(
    "rounds_df",
    help="location to read dataframe of cateorgorized teams from",
    type=str,
    nargs='?',
    const="annotated_results_df_final.pkl",
    default="annotated_results_df_final.pkl",
)
parser.add_argument(
    "team_df",
    help="location to dataframe of team information from",
    type=str,
    nargs='?',
    const="teams_scores_final.xlsx",
    default="teams_scores_final.xlsx",
)
parser.add_argument(
    "judge_df",
    help="location to read dataframe of judge information from",
    type=str,
    nargs='?',
    const="judge_scores_final.xlsx",
    default="judge_scores_final.xlsx",
)
parser.add_argument(
    "method",
    help="method to use in evaluating teamskill. options are 'team-wise', 'team-wise-sb', 'comprehensive', and 'TTT'.",
    type=str,
    nargs='?',
    const="team-wise",
    default="team-wise",
)
args = parser.parse_args()


def lc_to_dict(lc):
    lc_dict = {}
    for key, tuples_list in lc.items():
        sub_dict = {}
        for tuple_item in tuples_list:
            subkey, value = tuple_item
            sub_dict[subkey] = value
        lc_dict[key] = sub_dict
    return lc_dict

def annotate_skills(df_row, lc_dict):
    aff = df_row['aff']
    neg = df_row['neg']
    time = df_row['date'].timestamp()/(60*60*24)

    aff_mu = lc_dict[aff][time].mu
    aff_sigma = lc_dict[aff][time].sigma

    neg_mu = lc_dict[neg][time].mu
    neg_sigma = lc_dict[neg][time].sigma

    return aff_mu, aff_sigma, neg_mu, neg_sigma

def expected_aff_win(df_row):
    team_a = [ttt.Player(ttt.Gaussian(df_row['aff_mu'],df_row['aff_sigma'])) ]
    team_b = [ttt.Player(ttt.Gaussian(df_row['neg_mu'],df_row['neg_sigma'])) ]

    g = ttt.Game([team_a, team_b])
    return g.evidence


def mark_upsets(df_row, threshhold):
    if df_row['p_aff_win'] < threshhold and df_row['ballot'] == 1:
        return 1
    if df_row['p_aff_win'] > 1-threshhold and df_row['ballot'] == 0:
        return 1
    else:
        return 0


def get_history(matches, dates, results, eps=0.005, prior=None, sig=1, gam=0.01):
    # best sig, gams:
    # team focus: 1.859e-02  5.749e-03
    
    if not prior:
        prior = dict()
    print("Learning history...")
    h = ttt.History(composition=matches, results=results, priors=prior, times=dates, sigma=sig, gamma=gam)
    h.convergence(eps, iterations=100)
    return h


def chart_ttt(h, topcut=10):
    lc = h.learning_curves()
    sns.set_theme()
    fig, ax = plt.subplots(figsize=(8,6))
    team_df = pd.DataFrame(columns=['name','rating_max', 'rating_mean', 'rating_final', 'std_mean', 'last_round'])
    for i, agent in enumerate(h.agents.keys()):
        t = [v[0] for v in lc[agent]]
        mu = [v[1].mu for v in lc[agent]]
        sigma = [v[1].sigma for v in lc[agent]]
        last_round = datetime.fromtimestamp(lc[agent][-1][0]*60*60*24).strftime('%m/%d/%Y')
        team_df.loc[i] = [agent, max(mu), np.mean(mu),mu[-1], np.mean(sigma), last_round]
    top_df = team_df.sort_values('rating_max', ascending=False).iloc[0:topcut+1]
    for agent in top_df['name']:
        t = [datetime.fromtimestamp(v[0]*60*60*24) for v in lc[agent]]
        mu = [v[1].mu for v in lc[agent]]
        sigma = [v[1].sigma for v in lc[agent]]
        ax.plot(t, mu, label=agent)
        ax.fill_between(t, [m+s for m,s in zip(mu, sigma)], [m-s for m,s in zip(mu, sigma)], alpha=0.2) #type: ignore
    ax.legend()
    ax.set_title(f"Top {topcut} Teams' Learning Curves")
    ax.set_ylabel("Skill Distribution")
    ax.set_ylabel("")
    return team_df

def get_start_divs(uqteams, df):
    start_divs = []
    for team in uqteams:
        relevant = df[((df['aff'] == team)|(df['neg']==team))]
        start_div = relevant.sort_values('date')['division'].iloc[0]
        start_divs.append(start_div)
    return start_divs

#sigs 1.235e+00  1.529e+00  2.505e+00
#gam 5.289e-02 
#novmu, jvmu, opmu -1.592e+00, 0, 3.174e+00
def get_priors(uqteams, start_divs, sigs=[1.235e+00, 1.529e+00, 2.505e+00], gam= 5.289e-02, novmu=-1.592e+00, openmu=3.174e+00, beta=1):
    priors = dict()
    for team, start_div in zip(uqteams, start_divs):
        if start_div == 'novice':
            mu = novmu
            sig = sigs[0]
        elif start_div == 'jv':
            mu = 0.
            sig = sigs[1]
        else:
            mu = openmu
            sig = sigs[2]
        priors[team] = ttt.Player(ttt.Gaussian(mu, sig), beta, gam)
    return priors


def load_data(rounds_df, teams_df, judge_df):
    df = pd.read_pickle(rounds_df).reset_index().drop(columns='index')
    team_info = pd.read_excel(teams_df, index_col=0)
    judge_info = pd.read_excel(judge_df, index_col=0)

    # 2 is policy. 1 is flex. 0 is K. edit if needed
    num_to_word_t = {0:'p', 1:'f', 2:'k'}
    num_to_word_j = {0:'p', 1:'c', 2:'k'}

    team_info['assignment'] = team_info['naive pred tw'].map(num_to_word_t)
    judge_info['assignment'] = judge_info['assignment'].map(num_to_word_j)

    df['aff_type'] = df['aff assignment'].map(num_to_word_t)
    df['neg_type'] = df['neg assignment'].map(num_to_word_t)
    df['judge_type'] = df['judge assignment'].map(num_to_word_j)

    df['round_type'] = df.apply(lambda x: f'{x.aff_type} v {x.neg_type}, {x.judge_type} judge', axis=1)
    return df

def get_skill(df, method='TTT'):
    matches = [ [[r.aff],[r.neg]] for i, r in df.iterrows() ] 
    uqteams = list(set( [a for teams in matches for team in teams for a in team] ) ) 
    startdivs = get_start_divs(uqteams, df)
    dates = [ t.timestamp()/(60*60*24) for t in df.date]
    results = [[1,0] if ballot == 1 else [0,1] for ballot in df.ballot]
    priors = get_priors(uqteams, startdivs)
    if method=='TTT':
        pass 
    elif method=='team-focus':
        matches = [] 
        for i, r in df.iterrows():
            if (r.division == 'open') and (r.aff_type in ['p','k']) and (r.neg_type in ['p','k']):
                matches.append([[r.aff, f'{r.aff_type} v {r.neg_type}'],[r.neg]])
            else:
                matches.append([[r.aff],[r.neg]])
        priors['p v p'] = ttt.Player(ttt.Gaussian(1.51103273e-01, 1.00000000e-06), 0, 2.12121210e-03)
        priors['k v k'] = ttt.Player(ttt.Gaussian(6.97276576e-02, 1.00000000e-06), 0, 2.12121210e-03)
        priors['p v k'] = ttt.Player(ttt.Gaussian(3.07014780e-02, 1.00000000e-06), 0, 2.12121210e-03)
        priors['k v p'] = ttt.Player(ttt.Gaussian(-1.24697249e-02, 1.00000000e-06), 0, 2.12121210e-03)
    elif method =='comprehensive':
        matches = []
        for i, r in df.iterrows():
            if (r.division == 'open') and (r.aff_type in ['p','k']) and (r.neg_type in ['p','k']):
                matches.append([[r.aff, f'{r.aff_type} v {r.neg_type}, {r.judge_type} judge'],[r.neg]])
            else:
                matches.append([[r.aff],[r.neg]])
        sig= 7.788e-02  
        gam= 2.851e-03
        priors['p v p, p judge'] = ttt.Player(ttt.Gaussian(0.12754167, sig), 0, gam)
        priors['k v k, p judge'] = ttt.Player(ttt.Gaussian(0.22942759, sig), 0, gam)
        priors['p v k, p judge'] = ttt.Player(ttt.Gaussian(0.16170306, sig), 0, gam)
        priors['k v p, p judge'] = ttt.Player(ttt.Gaussian(-0.174479, sig), 0, gam)
        priors['p v p, k judge'] = ttt.Player(ttt.Gaussian(0.17020312, sig), 0, gam)
        priors['k v k, k judge'] = ttt.Player(ttt.Gaussian(0.04833764, sig), 0, gam)
        priors['p v k, k judge'] = ttt.Player(ttt.Gaussian(-0.15555476, sig), 0, gam)
        priors['k v p, k judge'] = ttt.Player(ttt.Gaussian(0.22786883, sig), 0, gam)
        priors['p v p, c judge'] = ttt.Player(ttt.Gaussian(0.04753675, sig), 0, gam)
        priors['k v k, c judge'] = ttt.Player(ttt.Gaussian(0.09135217, sig), 0, gam)
        priors['p v k, c judge'] = ttt.Player(ttt.Gaussian(-0.09183645, sig), 0, gam)
        priors['k v p, c judge'] = ttt.Player(ttt.Gaussian(0.03246564, sig), 0, gam)
    elif method=='team-focus-sb': 
        matches = [] 
        for i, r in df.iterrows():
            if (r.division == 'open') and (r.aff_type in ['p','k']) and (r.neg_type in ['p','k']):
                if (r.aff_mu > 5.64) and (r.neg_mu > 5.64):
                    matches.append([[r.aff, f'{r.aff_type} v {r.neg_type}, top'],[r.neg]])
                elif (r.aff_mu > 4.55) and (r.neg_mu > 4.55):
                    matches.append([[r.aff, f'{r.aff_type} v {r.neg_type}, upper'],[r.neg]])
                elif (r.aff_mu > 3.54) and (r.neg_mu > 3.54):
                    matches.append([[r.aff, f'{r.aff_type} v {r.neg_type}, middle'],[r.neg]])
                elif (r.aff_mu > 2.37) and (r.neg_mu > 2.37):
                    matches.append([[r.aff, f'{r.aff_type} v {r.neg_type}, lower'],[r.neg]])
                else:
                    matches.append([[r.aff, f'{r.aff_type} v {r.neg_type}, bottom'],[r.neg]])
            else:
                matches.append([[r.aff],[r.neg]])
        priors['p v p, top'] = ttt.Player(ttt.Gaussian(1.51103273e-01, 5.00000000e-01), 0, 2.12121210e-03)
        priors['p v p, upper'] = ttt.Player(ttt.Gaussian(1.51103273e-01, 5.00000000e-01), 0, 2.12121210e-03)
        priors['p v p, middle'] = ttt.Player(ttt.Gaussian(1.51103273e-01, 5.00000000e-01), 0, 2.12121210e-03)
        priors['p v p, lower'] = ttt.Player(ttt.Gaussian(1.51103273e-01, 5.00000000e-01), 0, 2.12121210e-03)
        priors['p v p, bottom'] = ttt.Player(ttt.Gaussian(1.51103273e-01, 5.00000000e-01), 0, 2.12121210e-03)

        priors['k v k, top'] = ttt.Player(ttt.Gaussian(6.97276576e-02, 5.00000000e-01), 0, 2.12121210e-03)
        priors['k v k, upper'] = ttt.Player(ttt.Gaussian(6.97276576e-02, 5.00000000e-01), 0, 2.12121210e-03)
        priors['k v k, middle'] = ttt.Player(ttt.Gaussian(6.97276576e-02, 5.00000000e-01), 0, 2.12121210e-03)
        priors['k v k, lower'] = ttt.Player(ttt.Gaussian(6.97276576e-02, 5.00000000e-01), 0, 2.12121210e-03)
        priors['k v k, bottom'] = ttt.Player(ttt.Gaussian(6.97276576e-02, 5.00000000e-01), 0, 2.12121210e-03)

        priors['p v k, top'] = ttt.Player(ttt.Gaussian(3.07014780e-02, 5.00000000e-01), 0, 2.12121210e-03)
        priors['p v k, upper'] = ttt.Player(ttt.Gaussian(3.07014780e-02, 5.00000000e-01), 0, 2.12121210e-03)
        priors['p v k, middle'] = ttt.Player(ttt.Gaussian(3.07014780e-02, 5.00000000e-01), 0, 2.12121210e-03)
        priors['p v k, lower'] = ttt.Player(ttt.Gaussian(3.07014780e-02, 5.00000000e-01), 0, 2.12121210e-03)
        priors['p v k, bottom'] = ttt.Player(ttt.Gaussian(3.07014780e-02, 5.00000000e-01), 0, 2.12121210e-03)

        priors['k v p, top'] = ttt.Player(ttt.Gaussian(-1.24697249e-02, 5.00000000e-01), 0, 2.12121210e-03)
        priors['k v p, upper'] = ttt.Player(ttt.Gaussian(-1.24697249e-02, 5.00000000e-01), 0, 2.12121210e-03)
        priors['k v p, middle'] = ttt.Player(ttt.Gaussian(-1.24697249e-02, 5.00000000e-01), 0, 2.12121210e-03)
        priors['k v p, lower'] = ttt.Player(ttt.Gaussian(-1.24697249e-02, 5.00000000e-01), 0, 2.12121210e-03)
        priors['k v p, bottom'] = ttt.Player(ttt.Gaussian(-1.24697249e-02, 5.00000000e-01), 0, 2.12121210e-03)
    else:
        print(f"error: {method} is invalid method")
        return
    
    print('learning history')
    h = get_history(matches, dates, results, prior=priors, eps=0.005)
    lc_dict = lc_to_dict(h.learning_curves())
    print('creating team skill dataframe')
    skill_df = chart_ttt(h, topcut=8)
    skill_df['team type'] = skill_df['name'].map(dict(zip(df['aff'], df['aff assignment']))| dict(zip(df['neg'], df['neg assignment'])))
    skill_df.to_excel(f'{method}_team_skill_df.xlsx')

    print('annotating round records df with skill info')
    df[['aff_mu', "aff_sigma", "neg_mu", "neg_sigma"]] = df.apply(annotate_skills, lc_dict=lc_dict, axis='columns', result_type='expand')
    df['p_aff_win'] = df.apply(expected_aff_win, axis='columns')
    df['is_upset'] = df.apply(mark_upsets, threshhold=0.4, axis='columns')
    df['error'] = df.apply(lambda x: abs(x['ballot']- x['p_aff_win']), axis='columns')
    df.to_excel(f'{method}_round_records_df.xlsx')

def main(rounds_df, team_df, judge_df, method):
    print('loading data')
    df = load_data(rounds_df, team_df, judge_df)
    get_skill(df, method)

if __name__ == "__main__":
    main(args.rounds_df, args.team_df, args.judge_df, args.method)
