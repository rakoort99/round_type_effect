import pandas as pd
import ast
import torch
import pickle
from tqdm.notebook import tqdm
import argparse

parser = argparse.ArgumentParser("gen_model_inputs")
parser.add_argument(
    "data_type",
    help='which subset of data to generate model inputs for. options are "synthetic", "full", "prelims", or "open_prelims".',
    type=str,
    nargs='?',
    const="synthetic",
    default="synthetic",
)
parser.add_argument(
    "round_data",
    help=".xlsx file containing records of debate rounds",
    type=str,
    nargs='?',
    const="round_records_clean_final2.xlsx",
    default="round_records_clean_final2.xlsx",
)
parser.add_argument(
    "judge_data",
    help=".xlsx file containing records of judges",
    type=str,
    nargs='?',
    const="paradigms_final2.xlsx",
    default="paradigms_final2.xlsx",
)

args = parser.parse_args()


def get_pool_tensor(pool_list, njudges):
    def poolmaker_helper(indices, njudge):
        v = torch.zeros(njudge, dtype=torch.bool)
        v[indices] = 1
        return v

    tensor_list = [poolmaker_helper(i, njudges) for i in pool_list]

    tensor = torch.stack(tensor_list)

    return tensor


def loopy_get_pool_tensor(pool_scale_list, njudges):
    def poolmaker_helper(tups, njudge):
        pool_marker = torch.zeros(njudge, dtype=torch.bool)
        scales = torch.zeros(njudge, dtype=torch.float)
        for tup in tups:
            idx = int(tup[0])
            val = tup[1]
            pool_marker[idx] = 1
            scales[idx] = val
        return pool_marker, scales

    pools_tensor_list = [poolmaker_helper(i, njudges)[0] for i in pool_scale_list]
    pools_tensor = torch.stack(pools_tensor_list)

    vals_tensor_list = [poolmaker_helper(i, njudges)[1] for i in pool_scale_list]
    vals_tensor = torch.stack(vals_tensor_list)

    return pools_tensor, vals_tensor


def full_round(round_records_df, judges_df):
    """creates model inputs using each round from dataframe of debate rounds
    and dataframe of judges"""

    print("loading data")
    results_df = pd.read_excel(round_records_df, index_col=0)
    results_df["name"] = results_df["name"].apply(lambda x: str(x).lower())
    results_df.sort_values("ID", inplace=True)

    paradigms_df = pd.read_excel(judges_df, index_col=0)

    print("processing data")
    # get dicts to translate into index names
    jname_to_jid = dict(zip(paradigms_df["Judge Name"], paradigms_df["Judge ID"]))
    jid_to_jindex = dict(zip(paradigms_df["Judge ID"], paradigms_df.index))
    jindex_to_jname = dict(zip(paradigms_df.index, paradigms_df["Judge Name"]))

    # translate judges into index names
    results_df["name"] = results_df["name"].apply(
        lambda x: jid_to_jindex[jname_to_jid[x]]
    )
    results_df["pool"] = results_df["pool"].apply(
        lambda x: [jid_to_jindex[jname_to_jid[i.lower()]] for i in ast.literal_eval(x)]
    )

    # translate teams into index names
    teams = pd.Series(
        pd.concat([results_df["aff"], results_df["neg"]], ignore_index=True).unique()
    )
    tname_to_tindex = dict(zip(teams.values, teams.index))
    tindex_to_tname = dict(zip(teams.index, teams.values))
    results_df["aff"] = results_df["aff"].apply(lambda x: tname_to_tindex[x])
    results_df["neg"] = results_df["neg"].apply(lambda x: tname_to_tindex[x])

    # reformat data
    judgelist = paradigms_df.index.values
    affs = torch.tensor(results_df["aff"].values)
    negs = torch.tensor(results_df["neg"].values)
    judges = torch.tensor(results_df["name"].values)
    teamlist = teams.index.values

    # get pools
    pool_list = results_df["pool"].to_list()
    pools = get_pool_tensor(pool_list, len(judgelist))

    # get roundwise pools
    rw_pools_dict = results_df.groupby(["ID", "round"])["name"].unique().to_dict()
    results_df["rw_pool"] = results_df.apply(
        lambda x: rw_pools_dict[(x["ID"], x["round"])].tolist(), axis=1
    )
    rw_pool_list = results_df["rw_pool"].to_list()
    rw_pools = get_pool_tensor(rw_pool_list, len(judgelist))

    print("saving data")
    # save affs, negs, judges, team_list, jude_list, and dicts
    torch.save(affs, "test/affs.pkl")
    torch.save(negs, "test/negs.pkl")
    torch.save(judges, "test/judges.pkl")
    torch.save(teamlist, "test/team_list.pkl")
    torch.save(judgelist, "test/judge_list.pkl")
    pickle.dump(tindex_to_tname, open("test/team_decoder.pkl", "wb"))
    pickle.dump(jindex_to_jname, open("test/judge_decoder.pkl", "wb"))
    torch.save(pools, "test/pools.pkl")
    torch.save(rw_pools, "test/rw_pools.pkl")
    print("done!")


def prelim_rounds(round_records_df, judges_df):
    ### Prelims Only
    print("loading data")
    results_df = pd.read_excel(round_records_df, index_col=0)
    results_df["name"] = results_df["name"].apply(lambda x: str(x).lower())

    prelims_df = results_df[results_df["panel"].isna()].copy(True)
    del results_df
    prelims_df = prelims_df[
        ~prelims_df["round"].isin(
            [
                "10R10",
                "9R9",
                "11Qtrs",
                "10Quarte",
                "7Octos",
                "10Sems",
                "8Quarte",
                "9Dbls",
                "11Finals",
                "10Octs",
                "12Final",
                "9Octs",
                "9Octos",
                "8OR 1",
                "7Nov Fi",
                "7Semis",
                "12Finals",
                "7Q",
                "8Sems",
            ]
        )
    ]

    # reduce to "non-tiny" tournaments
    real_ts = (
        prelims_df.groupby("ID")["aff"]
        .count()[prelims_df.groupby("ID")["aff"].count() > 50]
        .index
    )
    prelims_df = prelims_df[prelims_df["ID"].isin(real_ts)]

    prelims_df = prelims_df[~prelims_df["ID"].isin([1914, 1963])]

    # get guess for scaling factors based on judge availability
    rd_names = pd.read_excel("round_names.xlsx", index_col=0)
    messy_rd_to_clean = dict(zip(rd_names[0], rd_names[1]))
    prelims_df["round"] = prelims_df["round"].apply(lambda x: messy_rd_to_clean[x])

    prelims_df.sort_values("ID", inplace=True)

    paradigms_df = pd.read_excel(judges_df, index_col=0)
    print("processing data")
    # get dicts to translate into index names
    jname_to_jid = dict(zip(paradigms_df["Judge Name"], paradigms_df["Judge ID"]))
    jid_to_jindex = dict(zip(paradigms_df["Judge ID"], paradigms_df.index))
    jindex_to_jname = dict(zip(paradigms_df.index, paradigms_df["Judge Name"]))

    # translate judges into index names
    prelims_df["name"] = prelims_df["name"].apply(
        lambda x: jid_to_jindex[jname_to_jid[x]]
    )
    prelims_df["pool"] = prelims_df["pool"].apply(
        lambda x: [jid_to_jindex[jname_to_jid[i.lower()]] for i in ast.literal_eval(x)]
    )

    # translate teams into index names
    teams = pd.Series(
        pd.concat([prelims_df["aff"], prelims_df["neg"]], ignore_index=True).unique()
    )
    tname_to_tindex = dict(zip(teams.values, teams.index))
    tindex_to_tname = dict(zip(teams.index, teams.values))
    prelims_df["aff"] = prelims_df["aff"].apply(lambda x: tname_to_tindex[x])
    prelims_df["neg"] = prelims_df["neg"].apply(lambda x: tname_to_tindex[x])

    # reformat data
    judgelist = paradigms_df.index.values
    affs = torch.tensor(prelims_df["aff"].values)
    negs = torch.tensor(prelims_df["neg"].values)
    judges = torch.tensor(prelims_df["name"].values)
    teamlist = teams.index.values

    # get pools
    pool_list = prelims_df["pool"].to_list()
    pools = get_pool_tensor(pool_list, len(judgelist))

    # get dictionaries that allow us to form roundwise scaling factors
    nums_dict = {}
    denoms_dict = {}
    for i in range(1, 9):
        nums = prelims_df[prelims_df["round"] >= 1].groupby("ID")["name"].value_counts()
        key1 = [x[0] for x in nums.index]
        key2 = [x[1] for x in nums.index]
        key3 = [i for x in nums.index]
        vals = nums.values
        nums_dict = nums_dict | dict(zip(zip(key1, key2, key3), vals))

        denoms = prelims_df[prelims_df["round"] >= 1].groupby("ID")["name"].count()
        key1 = denoms.index
        key2 = [i for x in denoms.index]
        vals = denoms.values
        denoms_dict = denoms_dict | dict(zip(zip(key1, key2), vals))

    # check that dict is correct
    # if someone judges round 8, they are in the pool for all previous rounds
    for key in nums_dict.keys():
        rd = key[2]
        judge = key[1]
        tourn = key[0]
        if rd > 1:
            nums_dict[(tourn, judge, rd - 1)]
            assert nums_dict[(tourn, judge, rd - 1)] == nums_dict[(tourn, judge, rd)]

    # form df from dict that allows us to get pools

    nums_df = pd.DataFrame().from_dict(nums_dict, orient="index")
    nums_df.index = pd.MultiIndex.from_tuples(
        nums_df.index, names=["ID", "name", "round"]
    )
    nums_df.reset_index(inplace=True)
    nums_df["scale"] = nums_df.apply(
        lambda x: x[0] / denoms_dict[(x["ID"], x["round"])], axis=1
    )
    nums_df.drop(0, axis=1, inplace=True)
    nums_df["tuple"] = nums_df.apply(lambda x: (x["name"], x["scale"]), axis=1)
    # get roundwise pools and scales
    rw_pool_scale_dict = nums_df.groupby(["ID", "round"])["tuple"].apply(list).to_dict()

    prelims_df["rw_tuple"] = prelims_df.apply(
        lambda x: rw_pool_scale_dict[(x["ID"], x["round"])], axis=1
    )
    rw_pool_list = prelims_df["rw_tuple"].to_list()

    rw_pools, rw_scales = loopy_get_pool_tensor(rw_pool_list, len(judgelist))

    # check that pools, scales align
    assert (rw_pools * rw_scales != rw_scales).sum().item() == 0
    # check that each judge is properly in the pool
    for i, name in enumerate(prelims_df["name"]):
        check = rw_pools[i][name] is True
        assert check is True
    print("saving data")
    # save affs, negs, judges, team_list, jude_list, and dicts
    torch.save(affs, "test/pre_affs.pkl")
    torch.save(negs, "test/pre_negs.pkl")
    torch.save(judges, "test/pre_judges.pkl")
    torch.save(teamlist, "test/pre_team_list.pkl")
    torch.save(judgelist, "test/pre_judge_list.pkl")
    pickle.dump(tindex_to_tname, open("test/pre_team_decoder.pkl", "wb"))
    pickle.dump(jindex_to_jname, open("test/pre_judge_decoder.pkl", "wb"))
    torch.save(pools, "test/pre_pools.pkl")
    torch.save(rw_pools, "test/pre_rw_pools.pkl")
    torch.save(rw_scales, "test/pre_rw_scales.pkl")
    print("done!")


def open_prelim_rounds(round_records_df, judges_df):
    ### Open Prelims Only
    print("loading data")
    results_df = pd.read_excel(round_records_df, index_col=0)
    results_df["name"] = results_df["name"].apply(lambda x: str(x).lower())

    prelims_df = results_df[results_df["panel"].isna()].copy(True)
    prelims_df = prelims_df[
        ~prelims_df["round"].isin(
            [
                "10R10",
                "9R9",
                "11Qtrs",
                "10Quarte",
                "7Octos",
                "10Sems",
                "8Quarte",
                "9Dbls",
                "11Finals",
                "10Octs",
                "12Final",
                "9Octs",
                "9Octos",
                "8OR 1",
                "7Nov Fi",
                "7Semis",
                "12Finals",
                "7Q",
                "8Sems",
            ]
        )
    ]
    prelims_df = prelims_df[prelims_df["division"] == "open"]
    del results_df

    # get guess for scaling factors based on judge availability
    rd_names = pd.read_excel("round_names.xlsx", index_col=0)
    messy_rd_to_clean = dict(zip(rd_names[0], rd_names[1]))
    prelims_df["round"] = prelims_df["round"].apply(lambda x: messy_rd_to_clean[x])

    prelims_df.sort_values("ID", inplace=True)

    paradigms_df = pd.read_excel(judges_df, index_col=0)

    print("processing data")
    # get dicts to translate into index names
    jname_to_jid = dict(zip(paradigms_df["Judge Name"], paradigms_df["Judge ID"]))
    jid_to_jindex = dict(zip(paradigms_df["Judge ID"], paradigms_df.index))
    jindex_to_jname = dict(zip(paradigms_df.index, paradigms_df["Judge Name"]))

    # translate judges into index names
    prelims_df["name"] = prelims_df["name"].apply(
        lambda x: jid_to_jindex[jname_to_jid[x]]
    )
    prelims_df["pool"] = prelims_df["pool"].apply(
        lambda x: [jid_to_jindex[jname_to_jid[i.lower()]] for i in ast.literal_eval(x)]
    )

    # translate teams into index names
    teams = pd.Series(
        pd.concat([prelims_df["aff"], prelims_df["neg"]], ignore_index=True).unique()
    )
    tname_to_tindex = dict(zip(teams.values, teams.index))
    tindex_to_tname = dict(zip(teams.index, teams.values))
    prelims_df["aff"] = prelims_df["aff"].apply(lambda x: tname_to_tindex[x])
    prelims_df["neg"] = prelims_df["neg"].apply(lambda x: tname_to_tindex[x])

    # reformat data
    judgelist = paradigms_df.index.values
    affs = torch.tensor(prelims_df["aff"].values)
    negs = torch.tensor(prelims_df["neg"].values)
    judges = torch.tensor(prelims_df["name"].values)
    teamlist = teams.index.values

    # get pools
    pool_list = prelims_df["pool"].to_list()
    pools = get_pool_tensor(pool_list, len(judgelist))

    # get dictionaries that allow us to form roundwise scaling factors
    nums_dict = {}
    denoms_dict = {}
    for i in range(1, 9):
        nums = prelims_df[prelims_df["round"] >= 1].groupby("ID")["name"].value_counts()
        key1 = [x[0] for x in nums.index]
        key2 = [x[1] for x in nums.index]
        key3 = [i for x in nums.index]
        vals = nums.values
        nums_dict = nums_dict | dict(zip(zip(key1, key2, key3), vals))

        denoms = prelims_df[prelims_df["round"] >= 1].groupby("ID")["name"].count()
        key1 = denoms.index
        key2 = [i for x in denoms.index]
        vals = denoms.values
        denoms_dict = denoms_dict | dict(zip(zip(key1, key2), vals))

    # check that dict is correct
    # if someone judges round 8, they are in the pool for all previous rounds
    for key in nums_dict.keys():
        rd = key[2]
        judge = key[1]
        tourn = key[0]
        if rd > 1:
            nums_dict[(tourn, judge, rd - 1)]
            assert nums_dict[(tourn, judge, rd - 1)] == nums_dict[(tourn, judge, rd)]

    # form df from dict that allows us to get pools

    nums_df = pd.DataFrame().from_dict(nums_dict, orient="index")
    nums_df.index = pd.MultiIndex.from_tuples(
        nums_df.index, names=["ID", "name", "round"]
    )
    nums_df.reset_index(inplace=True)
    nums_df["scale"] = nums_df.apply(
        lambda x: x[0] / denoms_dict[(x["ID"], x["round"])], axis=1
    )
    nums_df.drop(0, axis=1, inplace=True)
    nums_df["tuple"] = nums_df.apply(lambda x: (x["name"], x["scale"]), axis=1)
    # get roundwise pools and scales
    rw_pool_scale_dict = nums_df.groupby(["ID", "round"])["tuple"].apply(list).to_dict()

    prelims_df["rw_tuple"] = prelims_df.apply(
        lambda x: rw_pool_scale_dict[(x["ID"], x["round"])], axis=1
    )
    rw_pool_list = prelims_df["rw_tuple"].to_list()

    ## made loopy function that outputs pools, scales given lists of tuples
    def loopy_get_pool_tensor(pool_scale_list, njudges):
        def poolmaker_helper(tups, njudge):
            pool_marker = torch.zeros(njudge, dtype=torch.bool)
            scales = torch.zeros(njudge, dtype=torch.float)
            for tup in tups:
                idx = int(tup[0])
                val = tup[1]
                pool_marker[idx] = 1
                scales[idx] = val
            return pool_marker, scales

        pools_tensor_list = [poolmaker_helper(i, njudges)[0] for i in pool_scale_list]
        pools_tensor = torch.stack(pools_tensor_list)

        vals_tensor_list = [poolmaker_helper(i, njudges)[1] for i in pool_scale_list]
        vals_tensor = torch.stack(vals_tensor_list)

        return pools_tensor, vals_tensor

    rw_pools, rw_scales = loopy_get_pool_tensor(rw_pool_list, len(judgelist))
    # check that pools, scales align
    assert (rw_pools * rw_scales != rw_scales).sum().item() == 0
    # check that each judge is properly in the pool
    for i, name in enumerate(prelims_df["name"]):
        check = rw_pools[i][name] is True
        assert check is True
    print("saving data")
    # save affs, negs, judges, team_list, jude_list, and dicts
    torch.save(affs, "test/op_pre_affs.pkl")
    torch.save(negs, "test/op_pre_negs.pkl")
    torch.save(judges, "test/op_pre_judges.pkl")
    torch.save(teamlist, "test/op_pre_team_list.pkl")
    torch.save(judgelist, "test/op_pre_judge_list.pkl")
    pickle.dump(tindex_to_tname, open("test/op_pre_team_decoder.pkl", "wb"))
    pickle.dump(jindex_to_jname, open("test/op_pre_judge_decoder.pkl", "wb"))
    torch.save(pools, "test/op_pre_pools.pkl")
    torch.save(rw_pools, "test/op_pre_rw_pools.pkl")
    torch.save(rw_scales, "test/op_pre_rw_scales.pkl")
    print("done!")


def synthetic_rounds(round_records_df, judges_df):
    ### Synthetic Data
    print("loading data")
    results_df = pd.read_excel(round_records_df, index_col=0)
    results_df["name"] = results_df["name"].apply(lambda x: str(x).lower())

    prelims_df = results_df[results_df["panel"].isna()].copy(True)
    prelims_df = prelims_df[
        ~prelims_df["round"].isin(
            [
                "10R10",
                "9R9",
                "11Qtrs",
                "10Quarte",
                "7Octos",
                "10Sems",
                "8Quarte",
                "9Dbls",
                "11Finals",
                "10Octs",
                "12Final",
                "9Octs",
                "9Octos",
                "8OR 1",
                "7Nov Fi",
                "7Semis",
                "12Finals",
                "7Q",
                "8Sems",
            ]
        )
    ]

    # reduce to "non-tiny" tournaments
    real_ts = (
        prelims_df.groupby("ID")["aff"]
        .count()[prelims_df.groupby("ID")["aff"].count() > 50]
        .index
    )
    prelims_df = prelims_df[prelims_df["ID"].isin(real_ts)]
    prelims_df.sort_values("ID", inplace=True)
    paradigms_df = pd.read_excel(judges_df, index_col=0)

    print("processing and generating data")
    # get dicts to translate into index names
    jname_to_jid = dict(zip(paradigms_df["Judge Name"], paradigms_df["Judge ID"]))
    jid_to_jindex = dict(zip(paradigms_df["Judge ID"], paradigms_df.index))
    dict(zip(paradigms_df.index, paradigms_df["Judge Name"]))

    # translate judges into index names
    prelims_df["name"] = prelims_df["name"].apply(
        lambda x: jid_to_jindex[jname_to_jid[x]]
    )
    prelims_df["pool"] = prelims_df["pool"].apply(
        lambda x: [jid_to_jindex[jname_to_jid[i.lower()]] for i in ast.literal_eval(x)]
    )

    # translate teams into index names
    teams = pd.Series(
        pd.concat([prelims_df["aff"], prelims_df["neg"]], ignore_index=True).unique()
    )
    tname_to_tindex = dict(zip(teams.values, teams.index))
    dict(zip(teams.index, teams.values))
    prelims_df["aff"] = prelims_df["aff"].apply(lambda x: tname_to_tindex[x])
    prelims_df["neg"] = prelims_df["neg"].apply(lambda x: tname_to_tindex[x])

    # reformat data
    judgelist = paradigms_df.index.values
    affs = torch.tensor(prelims_df["aff"].values)
    negs = torch.tensor(prelims_df["neg"].values)
    judges = torch.tensor(prelims_df["name"].values)
    teamlist = teams.index.values

    # get guess for scaling factors based on judge availability
    rd_names = pd.read_excel("round_names.xlsx", index_col=0)
    messy_rd_to_clean = dict(zip(rd_names[0], rd_names[1]))
    prelims_df["round"] = prelims_df["round"].apply(lambda x: messy_rd_to_clean[x])

    ndebates_r = prelims_df.groupby("ID").count()["season"]
    nrounds_r = prelims_df.groupby("ID")["round"].max()

    nteams_r = ndebates_r / nrounds_r * 2
    ntourn_r = ndebates_r.shape[0]
    njudges_r = prelims_df.groupby("ID")["pool"].max()
    njudges_r = njudges_r.apply(lambda x: len(ast.literal_eval(x))).to_list()

    def round_to_nearest_multiple_of_2(x):
        return round(x / 2) * 2

    nteams_r = nteams_r.apply(round_to_nearest_multiple_of_2)
    nteams_r = nteams_r.to_list()
    nrounds_r = nrounds_r.to_list()
    ndebates_r = ndebates_r.to_list()

    import numpy as np
    import scipy.stats as st

    nteams = len(teamlist)
    njudges = len(judgelist)

    # we generate 300 teams, 140 policy, 30 flex, 130 K
    team_idps = np.concatenate(
        [
            np.random.beta(13.38, 2.36, int(nteams / 15 * 7)),
            np.random.beta(28, 28, int(nteams / 10)),
            np.random.beta(2.36, 13.38, int(nteams / 30 * 13)),
        ],
        axis=0,
    )
    np.save("team_idps_fancy.pkl", team_idps, allow_pickle=True)

    # we generate 200 judges, 60 policy, 90 clash, 50 K
    judge_idps = np.concatenate(
        [
            np.random.beta(13.38, 2.36, int(njudges / 10 * 3)),
            np.random.beta(28, 28, int(njudges / 20 * 9)),
            np.random.beta(2.36, 13.38, int(njudges / 4)),
        ],
        axis=0,
    )

    np.save("judge_idps_fancy.pkl", judge_idps, allow_pickle=True)

    # we generate 'true' prefs : distance from team_idp to judge_idp, 300 x 200
    prefs = np.abs(
        team_idps[:, np.newaxis]
        - judge_idps[np.newaxis, :]
        + np.random.normal(0, 0.075 * np.ones((nteams, njudges)))
    )

    # simulated annealing to opimize stuff
    def hajel_sa_expon(x0, h, nchoices, N=10**4, c=4.0, freeze=100):
        X = x0
        hX = h(X[:nchoices])
        trumin = X
        hTrumin = hX

        freezecount = 0

        its = 0
        for i in range(N):
            if freezecount >= freeze:
                break
            its += 1

            Z = X.copy()
            random_swap1 = np.random.randint(0, nchoices)
            random_swap2 = np.random.randint(0, len(X))
            Z[random_swap1] = X[random_swap2]
            Z[random_swap2] = X[random_swap1]

            hZ = h(Z[:nchoices])
            Tn = c / np.log(i + 2)
            p = min(1, np.exp(-(hZ - hX) / Tn))
            if np.exp(-(hZ - hX) / Tn) == np.nan:
                print(hZ, hX, Tn)

            if st.uniform.rvs() < p:
                freezecount = 0

                X = Z.copy()
                hX = hZ
                if hZ < hTrumin:
                    trumin = Z
                    hTrumin = hZ
            else:
                freezecount += 1

        return X, trumin, its

    # function to assign judges give affs, negs, prefs
    def assign_judges(affs, negs, judges, tourn_prefs, availability):
        aff_prefs = tourn_prefs[affs, :]
        neg_prefs = tourn_prefs[negs, :]
        # print('aff, neg prefs')
        # print(aff_prefs)
        # print(neg_prefs)
        pref = aff_prefs + neg_prefs
        mut = np.abs(aff_prefs - neg_prefs)
        loss = 20 * pref + 40 * mut

        # giga penalty for judges out of the pool
        loss = loss + 1e6 * (availability == 0).astype(float)

        # print("loss")
        # print(loss)
        def get_loss(perm):
            return np.sum(loss[np.arange(len(perm)), perm])

        first_pass = []
        for i in loss:
            reuse_penalty = np.zeros_like(i)
            reuse_penalty[first_pass] = 10e5
            first_pass.append(np.argmin(i + reuse_penalty))
        # print('first pass and loss')
        # print(first_pass)
        # print(get_loss(first_pass))
        padded_first_pass = first_pass.copy()
        for i in range(len(judges)):
            if i not in first_pass:
                padded_first_pass.append(i)
        _, padded_best_solution, _ = hajel_sa_expon(
            padded_first_pass, get_loss, len(affs)
        )
        best_solution = padded_best_solution[: len(affs)]
        return judges[best_solution]

    # generate some pairings

    synthetic_df = pd.DataFrame()
    ntourn = ntourn_r
    nteams_r
    idslist = []
    affslist = []
    negslist = []
    roundlist = []
    judgelist = []
    for i in tqdm(range(ntourn)):
        attending_teams = np.random.permutation(nteams)[: nteams_r[i]]
        nassignments = int(nrounds_r[i] * nteams_r[i] / 2)
        attending_judges = np.random.permutation(njudges)[: njudges_r[i]]
        # draw judge availability
        p = 0.65
        availability = np.random.binomial(nrounds_r[i], p, len(attending_judges))
        while availability.sum() < nassignments:
            p += 0.05
            availability = np.random.binomial(nrounds_r[i], 0.65, len(attending_judges))
        available_rounds = availability.sum()
        pref_scales = availability / available_rounds
        scale_grid = np.tile(pref_scales, (len(attending_teams), 1))

        sort = np.argsort(prefs[attending_teams, :][:, attending_judges], axis=1)
        unsort = np.argsort(sort, axis=1)
        sorted_tournpref = scale_grid[
            np.arange(scale_grid.shape[0])[:, np.newaxis], sort
        ].cumsum(axis=1)
        tourn_prefs = sorted_tournpref[
            np.arange(scale_grid.shape[0])[:, np.newaxis], unsort
        ]

        for j in range(1, 1 + nrounds_r[i]):
            pairings = np.random.permutation(len(attending_teams))
            affs = pairings[: int(nteams_r[i] / 2)]
            negs = pairings[int(nteams_r[i] / 2) :]
            judge_assigns = assign_judges(
                affs, negs, attending_judges, tourn_prefs, availability
            )
            assert affs.shape == negs.shape
            availability[np.where(np.isin(judge_assigns, attending_judges))] += -1

            for k in range(len(affs)):
                idslist.append(i)
                affslist.append(attending_teams[affs[k]])
                negslist.append(attending_teams[negs[k]])
                roundlist.append(j)
                judgelist.append(judge_assigns[k])

    synthetic_df["ID"] = idslist
    synthetic_df["aff"] = affslist
    synthetic_df["neg"] = negslist
    synthetic_df["round"] = roundlist
    synthetic_df["name"] = judgelist

    ### Synthetic Only

    synthetic_df.sort_values("ID", inplace=True)

    # reformat data
    judgelist = np.arange(njudges)
    affs = torch.tensor(synthetic_df["aff"].values)
    negs = torch.tensor(synthetic_df["neg"].values)
    judges = torch.tensor(synthetic_df["name"].values)
    teamlist = np.arange(nteams)

    # get dictionaries that allow us to form roundwise scaling factors
    nums_dict = {}
    denoms_dict = {}
    for i in range(1, 9):
        nums = (
            synthetic_df[synthetic_df["round"] >= 1]
            .groupby("ID")["name"]
            .value_counts()
        )
        key1 = [x[0] for x in nums.index]
        key2 = [x[1] for x in nums.index]
        key3 = [i for x in nums.index]
        vals = nums.values
        nums_dict = nums_dict | dict(zip(zip(key1, key2, key3), vals))

        denoms = synthetic_df[synthetic_df["round"] >= 1].groupby("ID")["name"].count()
        key1 = denoms.index
        key2 = [i for x in denoms.index]
        vals = denoms.values
        denoms_dict = denoms_dict | dict(zip(zip(key1, key2), vals))

    # check that dict is correct
    # if someone judges round 8, they are in the pool for all previous rounds
    for key in nums_dict.keys():
        rd = key[2]
        judge = key[1]
        tourn = key[0]
        if rd > 1:
            nums_dict[(tourn, judge, rd - 1)]
            assert nums_dict[(tourn, judge, rd - 1)] == nums_dict[(tourn, judge, rd)]

    # form df from dict that allows us to get pools

    nums_df = pd.DataFrame().from_dict(nums_dict, orient="index")
    nums_df.index = pd.MultiIndex.from_tuples(
        nums_df.index, names=["ID", "name", "round"]
    )
    nums_df.reset_index(inplace=True)
    nums_df["scale"] = nums_df.apply(
        lambda x: x[0] / denoms_dict[(x["ID"], x["round"])], axis=1
    )

    nums_df.drop(0, axis=1, inplace=True)
    nums_df["tuple"] = nums_df.apply(lambda x: (x["name"], x["scale"]), axis=1)
    # get roundwise pools and scales
    rw_pool_scale_dict = nums_df.groupby(["ID", "round"])["tuple"].apply(list).to_dict()

    synthetic_df["rw_tuple"] = synthetic_df.apply(
        lambda x: rw_pool_scale_dict[(x["ID"], x["round"])], axis=1
    )
    rw_pool_list = synthetic_df["rw_tuple"].to_list()

    ## made loopy function that outputs pools, scales given lists of tuples
    def loopy_get_pool_tensor(pool_scale_list, njudges):
        def poolmaker_helper(tups, njudge):
            pool_marker = torch.zeros(njudge, dtype=torch.bool)
            scales = torch.zeros(njudge, dtype=torch.float)
            for tup in tups:
                idx = int(tup[0])
                val = tup[1]
                pool_marker[idx] = 1
                scales[idx] = val
            return pool_marker, scales

        pools_tensor_list = [poolmaker_helper(i, njudges)[0] for i in pool_scale_list]
        pools_tensor = torch.stack(pools_tensor_list)

        vals_tensor_list = [poolmaker_helper(i, njudges)[1] for i in pool_scale_list]
        vals_tensor = torch.stack(vals_tensor_list)
        return pools_tensor, vals_tensor

    rw_pools, rw_scales = loopy_get_pool_tensor(rw_pool_list, len(judgelist))
    # check that pools, scales align
    assert (rw_pools * rw_scales != rw_scales).sum().item() == 0
    # check that each judge is properly in the pool
    for i, name in enumerate(synthetic_df["name"]):
        check = rw_pools[i][name] is True
        assert check is True
    print("saving data")
    # save affs, negs, judges, team_list, jude_list, and dicts
    torch.save(affs, "test/toy_affs.pkl")
    torch.save(negs, "test/toy_negs.pkl")
    torch.save(judges, "test/toy_judges.pkl")
    torch.save(teamlist, "test/toy_team_list.pkl")
    torch.save(judgelist, "test/toy_judge_list.pkl")
    torch.save(rw_pools, "test/toy_rw_pools.pkl")
    torch.save(rw_scales, "test/toy_rw_scales.pkl")
    print("done!")


def main(round_records_df, judges_df, data_type):
    """generates full, prelim, open-prelim, or synthetic model inputs.
    saves them in "test" folder in cwd
    """
    if data_type=='full':
        print("processing full data")
        full_round(round_records_df, judges_df)
    elif data_type=='prelims':
        print("processing prelim data")
        prelim_rounds(round_records_df, judges_df)
    elif data_type=='open_prelims':
        print("processing open prelims data")
        open_prelim_rounds(round_records_df, judges_df)
    elif data_type=='synthetic':
        print("generating synthetic data")
        synthetic_rounds(round_records_df, judges_df)
    else:
        print(f'error: "{args.data_type}" is invalid data type')

if __name__ == "__main__":
    main(args.round_data, args.judge_data, args.data_type)
