import torch
import pyro
import numpy as np
import pandas as pd
import scipy.stats as ss
import argparse

parser = argparse.ArgumentParser("process_model")
parser.add_argument(
    "model_read",
    help="location to read model output from",
    type=str,
    nargs='?',
    const="models/op_pre_rw_1_warmup8000_batch20000.torch",
    default="models/op_pre_rw_1_warmup8000_batch20000.torch",
)
parser.add_argument(
    "rounds_read",
    help="location to read round records from",
    type=str,
    nargs='?',
    const="data/round_records_clean_final2.xlsx",
    default="data/round_records_clean_final2.xlsx",
)
parser.add_argument(
    "judges_read",
    help="location to read judge records from",
    type=str,
    nargs='?',
    const="data/paradigms_final2.xlsx",
    default="data/paradigms_final2.xlsx",
)
parser.add_argument(
    "rounds_write",
    help="location to write annotated round records to",
    type=str,
    nargs='?',
    const="annotated_results_df_final.pkl",
    default="annotated_results_df_final.pkl",
)
parser.add_argument(
    "teams_write",
    help="location to write team ideal point info to",
    type=str,
    nargs='?',
    const="teams_scores_final.xlsx",
    default="teams_scores_final.xlsx",
)
parser.add_argument(
    "judges_write",
    help="location to write judge ideal point info to",
    type=str,
    nargs='?',
    const="judge_scores_final.xlsx",
    default="judge_scores_final.xlsx",
)
args = parser.parse_args()


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.set_default_device(device)


def get_measures_t(row):
    a = float(row[0])
    b = float(row[1])
    dist = ss.beta(a, b)
    lower_mid = 0.333333
    upper_mid = 0.666667
    measures = np.array(
        [
            dist.cdf(lower_mid),
            dist.cdf(upper_mid) - dist.cdf(lower_mid),
            1 - dist.cdf(upper_mid),
        ]
    )
    return measures


def categorize_t(measures):
    assignment = np.argmax(measures)
    return assignment


def get_measures_j(row):
    a = float(row[0])
    b = float(row[1])
    dist = ss.beta(a, b)
    lower_mid = 0.333
    upper_mid = 0.667
    measures = np.array(
        [
            dist.cdf(lower_mid),
            dist.cdf(upper_mid) - dist.cdf(lower_mid),
            1 - dist.cdf(upper_mid),
        ]
    )
    return measures


def categorize_j(measures):
    assignment = np.argmax(measures)
    return assignment


def main(model_read, rounds_read, judges_read, rounds_write, teams_write, judges_write):
    ## load all data
    print('loading data')
    pyro.get_param_store().load(model_read, map_location=torch.device(device))
    results_df = pd.read_excel(rounds_read, index_col=0)
    paradigms_df = pd.read_excel(judges_read, index_col=0)

    team_concentrations = torch.stack([pyro.param("team_a"), pyro.param("team_b")]).T
    judge_concentrations = torch.stack([pyro.param("judge_a"), pyro.param("judge_b")]).T

    ## melt it down
    results_df = results_df.drop(
        ["round", "panel", "aff_last_div", "neg_last_div", "squirrel", "pool"], axis=1
    )
    results_df["name"] = results_df["name"].apply(lambda x: str(x).lower())

    # get dicts to translate index names
    jname_to_jid = dict(zip(paradigms_df["Judge Name"], paradigms_df["Judge ID"]))
    jid_to_jindex = dict(zip(paradigms_df["Judge ID"], paradigms_df.index))
    jindex_to_jname = dict(zip(paradigms_df.index, paradigms_df["Judge Name"]))

    # translate judges into index names
    results_df["name"] = results_df["name"].apply(
        lambda x: jid_to_jindex[jname_to_jid[x]]
    )

    # translate teams into index names
    teams = pd.Series(
        pd.concat([results_df["aff"], results_df["neg"]], ignore_index=True).unique()
    )
    tindex_to_tname = dict(zip(teams.index, teams.values))

    negcounts = results_df["neg"].value_counts()
    affcounts = results_df["aff"].value_counts()

    idx = affcounts.index.union(negcounts.index)
    out = (
        affcounts.reindex(idx)
        .fillna(0)
        .add(negcounts.reindex(idx).fillna(0))
        .astype(int)
    )
    tname_to_nrounds = out.to_dict()

    judgecounts = results_df["name"].value_counts()
    id_to_counts = judgecounts.to_dict()

    print('processing judges')
    ## Get judge scores
    judge_scores_df = pd.DataFrame(judge_concentrations.cpu().detach().numpy())
    judge_scores_df["name"] = judge_scores_df.index.map(jindex_to_jname.get)
    judge_scores_df["nrounds"] = judge_scores_df.index.map(id_to_counts.get)
    judge_scores_df["mean_0"] = judge_scores_df[0] / (
        judge_scores_df[0] + judge_scores_df[1]
    )
    judge_scores_df["var"] = (judge_scores_df[0] * judge_scores_df[1]) / (
        (judge_scores_df[0] + judge_scores_df[1]) ** 2
        * (judge_scores_df[0] + judge_scores_df[1] + 1)
    )
    judge_scores_df["measures"] = judge_scores_df.apply(get_measures_j, axis=1)
    judge_scores_df["assignment"] = judge_scores_df["measures"].apply(categorize_j)
    judge_scores_df.rename(
        columns={"name": "judge name", "nrounds": "rounds"}, inplace=True
    )
    jname_to_jass = dict(
        zip(
            judge_scores_df["judge name"].to_list(),
            judge_scores_df["assignment"].to_list(),
        )
    )

    ## assign to dataframe
    results_df["judge assignment"] = results_df["name"].apply(
        lambda x: jname_to_jass[jindex_to_jname[x]]
    )
    results_df["name"] = results_df["name"].apply(lambda x: jindex_to_jname[x])

    print('processing teams')
    ## get team scores
    concentrations_df = pd.DataFrame(team_concentrations.detach().cpu().numpy())
    concentrations_df["sum"] = concentrations_df.sum(1)
    concentrations_df["name"] = concentrations_df.index.map(tindex_to_tname.get)
    concentrations_df["rounds"] = concentrations_df["name"].map(tname_to_nrounds.get)
    concentrations_df["mean_0"] = concentrations_df[0] / (
        concentrations_df[0] + concentrations_df[1]
    )
    concentrations_df["var"] = (concentrations_df[0] * concentrations_df[1]) / (
        (concentrations_df[0] + concentrations_df[1]) ** 2
        * (concentrations_df[0] + concentrations_df[1] + 1)
    )
    concentrations_df["measures"] = concentrations_df.apply(get_measures_t, axis=1)
    concentrations_df["assignment"] = concentrations_df["measures"].apply(categorize_t)
    concentrations_df.rename(columns={"name": "team name"}, inplace=True)

    ## classify teams based on who judged them
    tourny_base = (
        results_df[["ID", "judge assignment"]].groupby("ID").value_counts()
        / results_df["ID"].value_counts().sort_index()
    )
    prob_predicts_tw = []
    for team in teams:
        relevant_rounds = results_df[
            (results_df["aff"] == team) | (results_df["neg"] == team)
        ]
        tourny_mix = relevant_rounds["ID"].value_counts() / relevant_rounds.shape[0]
        relevant_assignments = relevant_rounds[["ID", "judge assignment"]].groupby("ID")
        probs = (
            relevant_assignments.value_counts()
            / relevant_rounds["ID"].value_counts().sort_index()
            * tourny_mix
        )
        probs = probs.sort_index()
        realtive_probs = (probs / tourny_base).groupby("judge assignment").sum()
        idxmax = realtive_probs.argmax()
        idxmin = realtive_probs.argmin()
        # fuzz for neutrality
        # neutral if less likely than average to get both K and Policy
        if (realtive_probs[0] < 0.9) and (realtive_probs[2] < 0.9):
            prob_predicts_tw.append(1)
        # neutral if odds are the same as the aggregate (don't care about ideology in prefs)
        elif np.abs(realtive_probs[0] - realtive_probs[2]) < 0.15:
            prob_predicts_tw.append(1)
        # if not neutral, base off most/least likely pref
        elif (idxmax == 2) or (idxmin == 0):
            prob_predicts_tw.append(2)
        elif (idxmax == 0) or (idxmin == 2):
            prob_predicts_tw.append(0)
        # redundancy
        else:
            print("wtf on aisle", team)
            prob_predicts_tw.append(1)

    print("finishing up")
    ## assign to df
    tname_to_pred = dict(zip(teams, prob_predicts_tw))
    concentrations_df["naive pred tw"] = concentrations_df["team name"].apply(
        lambda x: tname_to_pred[x]
    )

    ## annotate categories to results_df
    results_df["aff assignment"] = results_df["aff"].map(tname_to_pred)
    results_df["neg assignment"] = results_df["neg"].map(tname_to_pred)
    results_df["team cats"] = results_df.apply(
        lambda x: set([x["aff assignment"], x["neg assignment"]]), axis=1
    )

    results_df.to_pickle(rounds_write)
    concentrations_df.sort_index().to_excel(teams_write)
    judge_scores_df.sort_index().to_excel(judges_write)


if __name__ == "__main__":
    main(
        args.model_read,
        args.rounds_read,
        args.judges_read,
        args.rounds_write,
        args.teams_write,
        args.judges_write,
    )
