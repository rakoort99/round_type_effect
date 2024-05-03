from datetime import datetime
import pandas as pd
import ast
import argparse

parser = argparse.ArgumentParser("gen_model_inputs")
parser.add_argument(
    "input",
    help=".xlsx file containing dirty records of debate rounds",
    type=str,
    nargs='?',
    const="round_records_final2.xlsx",
    default="round_records_final2.xlsx",
)
parser.add_argument(
    "output",
    help=".xlsx file to write clean records to",
    type=str,
    nargs='?',
    const="round_records_clean_final2.xlsx",
    default="round_records_clean_final2.xlsx",
)
args = parser.parse_args()


def remove_useless(dt):
    """removes useless column, "bad" tournaments from dataset"""
    data = dt.copy()
    # this was to represent that these are college debates. useless, they all are.
    data.drop("?", axis=1, inplace=True)

    # Drop practice/mislabeled tournaments
    data = data[not data["tourn"].str.contains("Institute")]
    data = data[not data["tourn"].str.contains("Practice")]

    # Drop non-policy tournaments
    data = data[not data["tourn"].str.contains("NCFA Spring Championship")]
    return data


def fix_panel(dt):
    """fixes panel result to reflect aff (1) or neg (0) ballot"""
    data = dt.copy()
    data["panel"] = data["panel"].apply(
        lambda x: str(x).lower()[:3] if isinstance(x, str) else x
    )

    def check(x):
        if isinstance(x, str):
            if x == "aff":
                return int(1)
            return int(0)
        return x

    data["panel"] = data["panel"].apply(check)
    return data


def fix_div(dt):
    """standardizes division names"""
    data = dt.copy()
    # set all division names lowercase
    data["division"] = data["division"].apply(lambda x: str(x).lower())

    # remove parli, ld, pf from rows
    data = data[not data["division"].str.contains("par")]
    data = data[not data["division"].str.contains("ld")]
    data = data[not data["division"].str.contains("pf")]

    # remove random big tent online events
    data = data[not data["division"].str.contains("pda")]
    data = data[not data["division"].str.contains("card")]
    data = data[not data["division"].str.contains("crd")]

    data = data[
        (
            (data["division"] == "nan")
            & (data["tourn"] == "University of Minnesota College Invitational")
        )
        is False
    ]

    # drop pnw debate
    data = data[not data["division"].str.contains("pnw")]

    # drop rookie debate
    data = data[not data["division"].str.contains("rook")]
    data = data[not data["division"].str.contains("rcx")]

    # fix some misnamed events
    data["division"].mask(data["division"] == "a name", "novice", inplace=True)
    data["division"].mask(data["division"] == "nwced", "open", inplace=True)

    # fix round robins
    data["division"].mask(data["division"].str.contains("rr"), "open", inplace=True)

    # now that the chaff is culled, broad fixing
    data["division"].mask(data["division"].str.contains("op"), "open", inplace=True)
    data["division"].mask(data["division"].str.contains("jv"), "jv", inplace=True)
    data["division"].mask(data["division"].str.contains("nov"), "novice", inplace=True)

    # remaining obvious specific instances
    data["division"].mask(
        # 45 different names for "open"....
        data["division"].isin(
            [
                "o",
                "v",
                "d1q",
                "qual",
                "nan",
                "cutt",
                "pitt",
                "q",
                "shir",
                "shirley",
                "cx var",
                "val",
                "txo",
                "ocx",
                "o pol",
                "o cx",
                "o-pers",
                "d7",
                "d5",
                "d4",
                "ndtd3",
                "var",
                "ndt",
                "cx-o",
                "oon",
                "d5qual",
                "d2",
                "d8",
                "mac",
                "d3",
                "v cx",
                "0",
                "quals",
                "d2ndt",
                "mac v",
                "oc",
                "-",
                "cuttr",
                "cx",
                "o-pol",
                "mac o",
                "mukai",
                "od",
                "brkot",
            ]
        ),
        "open",
        inplace=True,
    )
    data["division"].mask(
        data["division"].isin(
            ["jcx", "break", "jnr", "j", "pol", "njddt", "fyb", "junior"]
        ),
        "jv",
        inplace=True,
    )
    data["division"].mask(
        data["division"].isin(
            ["ncx", "nd", "mac n", "npol", "n cx", "cx-n", "lil 5", "n pol", "n", "np"]
        ),
        "novice",
        inplace=True,
    )
    return data


def fix_dates(dt):
    """cleans date column, converts to datetime"""
    data = dt.copy()
    # make date column play nice
    data["date"] = data["date"].apply(lambda x: x[-10:])
    data["date"] = pd.to_datetime(data["date"])
    return data


def fix_ballot(dt):
    """removes rows with invalid results, converts ballots to binary"""
    data = dt.copy()

    # remove invalid round results (caused by people dropping from tournament)
    data["ballot"] = data["ballot"].str.lower()
    data = data[data["ballot"].isin(["aff", "neg"])]

    # convert to binary output representing if aff wins
    data.ballot = data.ballot.apply(lambda x: 1 if x == "aff" else 0)
    return data


### component functions to fix team
def online_untagger(string):
    """removes online demarcations from teams"""
    onl_list = [" - ONLINE", " - Online", " ONLINE", " Online", " - ONL", " - Onl"]
    for tag in onl_list:
        if string[-len(tag) :] == tag:
            return string[: -len(tag)]
    return string


def unonewordifier(df):
    """normalizes a particular team name style"""
    newdf = df.copy()
    newdf["aff"] = newdf["aff"].apply(
        lambda x: (
            x[:-2] + " " + x[-2:]
            if len(x) > 3 and x[-2:].isupper() and x[-3] != " "
            else x
        )
    )
    newdf["neg"] = newdf["neg"].apply(
        lambda x: (
            x[:-2] + " " + x[-2:]
            if len(x) > 3 and x[-2:].isupper() and x[-3] != " "
            else x
        )
    )
    return newdf


def onewordify(string, td_dict):
    """converts teamcodes to one word"""
    words = string.split()
    for it in range(1, len(words)):
        firstpart = " ".join(words[:-it])
        lastpart = " ".join(words[-it:])
        try:
            newpart = td_dict[firstpart]
            return " ".join([newpart, lastpart])
        except:  # noqa: E722
            pass
    return string


def fix_amps(string):
    """removes ampersands from team codes"""
    if "&" in string:
        firsthalf = string.split("&")[0].strip()
        secondhalf = string.split("&")[1].strip()

        firstinits = "".join(word[0].upper() for word in firsthalf.split())
        secondinits = "".join(word[0].upper() for word in secondhalf.split())
        try:
            return (
                firsthalf.split(" ")[0].strip() + " " + firstinits[-1] + secondinits[-1]
            )
        except:  # noqa: E722
            print(string)
            return firsthalf.split(" ")[0].strip() + " " + firstinits[-1]
    elif " and " in string:
        firsthalf = string.split(" and ")[0].strip()
        secondhalf = string.split(" and ")[1].strip()

        firstinits = "".join(word[0].upper() for word in firsthalf.split())
        secondinits = "".join(word[0].upper() for word in secondhalf.split())
        try:
            return (
                firsthalf.split(" ")[0].strip() + " " + firstinits[-1] + secondinits[-1]
            )
        except:  # noqa: E722
            print(string)
            return firsthalf.split(" ")[0].strip() + " " + firstinits[-1]
    else:
        return string


def the_unscrambler(df):
    """reorders team codes to be consistent
    this can result in merging teams in edge cases, but is much more beneficial than harmful.
    we have unique IDs anyways - these are just the human-interpretable names
    """
    newdf = df.copy()
    newdf["aff"], newdf["neg"] = newdf["aff"].astype(str), newdf["neg"].astype(str)
    teams_accepted = set()
    rejects = set()
    reject_dict = {}
    new_aff = []
    new_neg = []
    for team in newdf.aff:
        if team in teams_accepted:
            new_aff.append(team)
        elif team in rejects:
            new_aff.append(reject_dict[team])
        elif team[:-2] + team[::-1][0:2] in teams_accepted:
            rejects.add(team)
            reject_dict[team] = team[:-2] + team[::-1][0:2]
            new_aff.append(reject_dict[team])
        else:
            teams_accepted.add(team)
            new_aff.append(team)
    for team in newdf.neg:
        if team in teams_accepted:
            new_neg.append(team)
        elif team in rejects:
            new_neg.append(reject_dict[team])
        elif team[:-2] + team[::-1][0:2] in teams_accepted:
            rejects.add(team)
            reject_dict[team] = team[:-2] + team[::-1][0:2]
            new_neg.append(reject_dict[team])
        else:
            teams_accepted.add(team)
            new_neg.append(team)
    newdf["aff"] = new_aff
    newdf["neg"] = new_neg
    return newdf


def fix_teams(dt):
    """applies all our teamcode fixing functions to aff,neg cols"""
    data = dt.copy()
    # dropnas
    data = data.dropna(subset=["aff"])
    data = data.dropna(subset=["neg"])

    # ensure type consistency, as early teams (2013) are randomly ints sometimes
    data["aff"] = data["aff"].astype(str)
    data["neg"] = data["neg"].astype(str)

    # get rid of ' - ONLINE' and '- ONL' suffixes
    data["aff"] = data["aff"].apply(online_untagger)
    data["neg"] = data["neg"].apply(online_untagger)

    # Add extra space when needed
    data = unonewordifier(data)

    # Remove unnecessary dashes and double spaces
    data["aff"] = data["aff"].str.replace(" - ", " ")
    data["aff"] = data["aff"].str.replace("  ", " ")
    data["neg"] = data["neg"].str.replace(" - ", " ")
    data["neg"] = data["neg"].str.replace("  ", " ")

    # stupid method to turn df to usable dict
    td = pd.read_excel("team_decoder.xlsx", index_col=0)
    td_dict = {}
    for idx in td.index:
        for val in td.loc[idx].dropna():
            td_dict[val] = idx

    # onewordify school names
    data["aff"] = data["aff"].apply(onewordify, td_dict=td_dict)
    data["neg"] = data["neg"].apply(onewordify, td_dict=td_dict)

    # remove ampersand teamcodes such that teams take form "School XY". This can cause problems, but
    # fixes vastly more than it causes. Trolley problem...
    data["aff"] = data["aff"].apply(fix_amps)
    data["neg"] = data["neg"].apply(fix_amps)

    # unscramble team codes, declaring "School XY" == "School YX". This can cause errors, but
    # fixes vastly more problems than it causes. Trolley problem...
    data = the_unscrambler(data)

    return data


def get_last_divs(data):
    """adds new columns for division each team competed in at their last tournament"""
    df = data.copy()
    df["date"] = pd.to_datetime(df["date"])

    def first_value_previous_date_aff(group):
        dt = group[["division", "aff", "date"]]
        last = pd.NA
        curr = pd.NA
        newcol = []
        for idx, (_, row) in enumerate(dt.iterrows()):
            if idx == 0:
                curr_time = row["date"]
            if row["date"] > curr_time:
                curr_time = row["date"]
                last = curr
            curr = row["division"]
            newcol.append(last)
        group["aff_prev_div"] = newcol
        return group["aff_prev_div"]

    def first_value_previous_date_neg(group):
        dt = group[["division", "neg", "date"]]
        last = pd.NA
        curr = pd.NA
        newcol = []
        for idx, (_, row) in enumerate(dt.iterrows()):
            if idx == 0:
                curr_time = row["date"]
            if row["date"] > curr_time:
                curr_time = row["date"]
                last = curr
            curr = row["division"]
            newcol.append(last)
        group["neg_prev_div"] = newcol
        return group["neg_prev_div"]

    # aff
    df.sort_values(["aff", "date"], inplace=True)
    aff_last_div = (
        df.groupby(["aff"]).apply(first_value_previous_date_aff).reset_index(drop=True)
    )
    df["aff_last_div"] = aff_last_div.values

    # neg
    df.sort_values(["neg", "date"], inplace=True)
    neg_last_div = (
        df.groupby(["neg"]).apply(first_value_previous_date_neg).reset_index(drop=True)
    )
    df["neg_last_div"] = neg_last_div.values

    return df


def mark_squirrel(df_row):
    if not pd.isna(df_row["panel"]):
        if df_row["panel"] - df_row["ballot"] == 0:
            return 0
        else:
            return 1
    else:
        return df_row["panel"]


def fix_uq_ids(data):
    df = data.copy()
    df["aff_id"] = df["aff_id"].apply(
        lambda x: frozenset(ast.literal_eval(x)) if pd.notna(x) else None
    )
    df["neg_id"] = df["neg_id"].apply(
        lambda x: frozenset(ast.literal_eval(x)) if pd.notna(x) else None
    )

    df.sort_values(by=["aff", "date"], inplace=True)
    df = df[~(df["aff_id"].isna() + df["neg_id"].isna())]
    # put together dict of id:name
    used_names = set()
    id_to_name = {}
    for i, row in df.iterrows():
        aff_id = row["aff_id"]
        neg_id = row["neg_id"]
        # only proceed if aff_id not None
        if aff_id:
            # only proceed if we don't have the id in dict
            if aff_id not in id_to_name:
                name = row["aff"]
                collision_count = 0
                suffix = ""
                name_candidate = name
                # add suffix if candidate name already in use
                while name_candidate in used_names:
                    collision_count += 1
                    suffix = f" {collision_count}"
                    name_candidate = name + suffix
                used_names.add(name_candidate)
                id_to_name[aff_id] = name_candidate
        if neg_id:
            # only proceed if we don't have the id in dict
            if neg_id not in id_to_name:
                name = row["neg"]
                collision_count = 0
                suffix = ""
                name_candidate = name
                # add suffix if candidate name already in use
                while name_candidate in used_names:
                    collision_count += 1
                    suffix = f" {collision_count}"
                    name_candidate = name + suffix
                used_names.add(name_candidate)
                id_to_name[neg_id] = name_candidate
    df["aff"] = df["aff_id"].apply(lambda x: id_to_name.get(x))
    df["neg"] = df["neg_id"].apply(lambda x: id_to_name.get(x))
    return df


def main(input_name, output_name):
    """fixes our round records"""
    print(f"Loading data. Time is now: {datetime.now()}")
    data = pd.read_excel(input_name, index_col=0)
    print("Starting size:", data.shape[0])
    print(f"Removing bad data. Time is now: {datetime.now()}")
    print(f"Cleaning columns. Time is now: {datetime.now()}")
    data = fix_dates(data)
    data = fix_teams(data)

    data = remove_useless(data)
    print("Size after removing non-policy tournaments", data.shape[0])
    data = fix_div(data)
    print("Size after removing non-policy rounds", data.shape[0])
    data = fix_ballot(data)
    data = fix_panel(data)
    data = fix_uq_ids(data)
    print("Size after removing unknown teams", data.shape[0])
    print(f"Adding finishing touches. Time is now: {datetime.now()}")
    data = get_last_divs(data)
    data["squirrel"] = data.apply(mark_squirrel, axis=1)
    print(f"Exporting data. Time is now: {datetime.now()}")
    data.to_excel(output_name)
    print(f"All done! Time is now: {datetime.now()}")


if __name__ == "__main__":
    main(args.input, args.output)
