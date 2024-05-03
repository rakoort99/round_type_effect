from collections import Counter
import datetime
import requests
from bs4 import BeautifulSoup
import openpyxl
import pandas as pd
from tqdm.auto import tqdm
import argparse

parser = argparse.ArgumentParser("gen_model_inputs")
parser.add_argument(
    "tourn_list",
    help=".xlsx file containing list of tournaments to scrape data from",
    type=str,
    nargs='?',
    const="tourn_list_final1.xlsx",
    default="tourn_list_final1.xlsx",
)
parser.add_argument(
    "judge_output",
    help=".xlsx file to write judge records to",
    type=str,
    nargs='?',
    const="paradigms_final2.xlsx",
    default="paradigms_final2.xlsx",
)
parser.add_argument(
    "round_output",
    help=".xlsx file to write round records to",
    type=str,
    nargs='?',
    const="round_records_final2.xlsx",
    default="round_records_final2.xlsx",
)

args = parser.parse_args()


class Judge:
    """judge class holds judge info"""

    def __init__(self, id_num, tid_dict):
        self.id_num = id_num
        self.url = (
            f"https://www.tabroom.com/index/paradigm.mhtml?judge_person_id={id_num}"
        )
        self.ceda_side_counter = Counter()

        try:
            # load judge page
            judge_page = requests.get(self.url, timeout=30)
            judge_souped = BeautifulSoup(judge_page.content, "html5lib")

            # record paradigm text, judge name
            try:
                self.para = judge_souped.find(
                    class_="paradigm ltborderbottom"
                ).get_text(strip=False, separator=" ")
            except Exception as e:
                print(f"Broken paradigm: {self.url}")
                self.para = ""
                print(e)
            self.name = judge_souped.h3.string

            # get round results info, iterate through to record
            round_rows = judge_souped.find("tbody").find_all("tr")  # type: ignore
            ceda_list = []
            for tr in round_rows:
                td = tr.find_all("td")
                row = [i for i in td]
                if len(row) == 0:
                    continue
                data = row[0].find("a")

                # get tournament ID from row of judge results table
                url = data["href"]
                marker = url.find("=")
                tid = url[marker + 1 :]

                # only proceed if it's a ceda round from tournament list
                if tid in tid_dict:
                    tourn = tid_dict.get(tid, "ignore")
                    clean_row = [i.get_text(strip=True) for i in row]

                    # get unique ids of participants. this is messy because the edge case
                    # requires going to a different page, unfortunately
                    team_dict = tourn.teams_dict
                    # aff
                    if clean_row[5] in team_dict:
                        uq_aff = team_dict[clean_row[5]]
                    else:
                        try:
                            res_page = requests.get(
                                f"https://www.tabroom.com{ row[5].find('a').get('href')}",
                                timeout=30,
                            )
                            res_soup = BeautifulSoup(res_page.content, "html5lib")
                            button = res_soup.find(
                                "a", attrs={"class": "buttonwhite thin greentext"}
                            )
                            ids = button.get("href").split("?")[-1].split("&")
                            uq_aff = set([i[4:] for i in ids])
                        except Exception as e:
                            uq_aff = None
                            print(
                                f"Error with team {clean_row[5]} at tournament {tourn, tid} with judge {self.name}"
                            )
                            print(e)

                    # neg
                    if clean_row[6] in team_dict:
                        uq_neg = team_dict[clean_row[6]]
                    else:
                        try:
                            res_page = requests.get(
                                f"https://www.tabroom.com{ row[6].find('a').get('href')}",
                                timeout=30,
                            )
                            res_soup = BeautifulSoup(res_page.content, "html5lib")
                            button = res_soup.find(
                                "a", attrs={"class": "buttonwhite thin greentext"}
                            )
                            ids = button.get("href").split("?")[-1].split("&")
                            uq_neg = set([i[4:] for i in ids])
                        except Exception as e:
                            uq_neg = None
                            print(
                                f"Error with team {clean_row[6]} at tournament {tourn, tid} with judge {self.name}"
                            )
                            print(e)
                    # year is empty string if tid not in dict, else it's Tournament class
                    # with year attribute
                    year = tourn.year
                    clean_row = [tid, year] + clean_row + [uq_aff, uq_neg]

                    # sort row into
                    ceda_list.append(clean_row)

            # record info
            self.ceda_round_list = ceda_list
            self.ceda_rounds = len(ceda_list)

        except Exception as e:
            print(f"Broken judge: {self.url}")
            self.para = ""
            self.name = ""
            self.ceda_round_list = []
            self.reject_list = []
            self.ceda_rounds = 0
            print(e)

    def __repr__(self):
        return f"{self.id_num}"

    def side_count(self):
        """used to figure out how many aff, neg round judged"""
        self.ceda_side_counter = Counter()
        for entry in self.ceda_round_list:
            self.ceda_side_counter[f"{entry[1]} {entry[9]}"] += 1
            self.ceda_side_counter[f"{entry[9]} Total"] += 1


class Tournament:
    """tournament class hold tournament info"""

    def __init__(self, name, url):
        self.name = name
        self.url = url
        equator = self.url.find("=")
        self.tourn_id = self.url[equator + 1 :]
        self.judge_url = (
            f"https://www.tabroom.com/index/tourn/judges.mhtml?tourn_id={self.tourn_id}"
        )
        self.year = 0
        self.events_url = (
            f"https://www.tabroom.com/index/tourn/fields.mhtml?tourn_id={self.tourn_id}"
        )
        self.teams_dict = {}
        try:
            # load events page
            events_page = requests.get(self.events_url, timeout=30)
            events_souped = BeautifulSoup(events_page.content, "html5lib")
            events_box = events_souped.find(
                "div", attrs={"class": "sidenote"}
            ).find_all("a", href=True)
            event_urls = []

            # iterate through judge pages different events at the tournament, record info
            for event in events_box:
                event_urls.append(f"https://www.tabroom.com{event.get('href')}")
            for url in event_urls:
                try:
                    event_page = requests.get(url, timeout=30)
                    event_souped = BeautifulSoup(event_page.content, "html5lib")
                    team_table = event_souped.find(id="fieldsort")
                    team_rows = team_table.find_all("tr")
                    for row in team_rows[1:]:
                        info = row.find_all("td")
                        team_name = info[3].text.strip()
                        ids = (
                            info[-1]
                            .find("a", href=True)
                            .get("href")
                            .split("?")[-1]
                            .split("&")
                        )
                        ids = set([i[4:] for i in ids])
                        self.teams_dict[team_name] = ids
                except Exception as e:
                    print(f"Error with event page: {url}")
                    print(e)

        except Exception as e:
            print(f"Error with events page: {self.events_url}")
            print(e)

    def __repr__(self):
        return f"{self.name}"


def load_tourney_list(file):
    """opens tournament list as dictionary mapping year to list of Tournament objs"""
    wb = openpyxl.load_workbook(file)
    lookup_dict = {}
    for sheet in wb:
        tourn_list = [
            Tournament(row[0].value, row[0].hyperlink.target) for row in sheet
        ]
        lookup_dict.update({sheet.title: tourn_list})

    return lookup_dict


def judge_cat_finder(input_url):
    """gets list of category judge pages from tournament judge page"""
    return_info = []
    try:
        list_info = requests.get(input_url, timeout=30)
    except Exception as e:
        print(f"Error with tournamnent page: {input_url}")
        print(e)
        return return_info

    souped_info = BeautifulSoup(list_info.content, "html5lib")
    cat_lists = souped_info.find_all("a")
    for cat in cat_lists:
        if not cat.get("href"):
            pass
        elif "judges.mhtml?category_id" in cat.get("href"):
            return_info.append(f"https://www.tabroom.com/{cat.get('href')}")

    return return_info


# note: there are lots of broken category pages. errors happening here are
# largely because of blank pages for cancelled tourneys, fake events. not something
# to worry about!
def judge_info_finder(category_lists, judge_dict, tid_dict):
    """modifies global judgdict to include Judge objects for all judges in
    tournament judge categories"""
    for url in category_lists:
        try:
            category_page = requests.get(url, timeout=30)
            cat_info = BeautifulSoup(category_page.content, "html5lib")
            cat_table = cat_info.find(id="judgelist")
            cat_rows = cat_table.find_all("td")
            for x in cat_rows:
                para_find = x.find_all("a", href=True)
                if len(para_find) > 0:
                    paradig = para_find[0].get("href")
                    equal_num = paradig.find("=")
                    judge_id = paradig[equal_num + 1 :]
                    if judge_id not in judge_dict:
                        judge_dict.update({judge_id: Judge(judge_id, tid_dict)})
        except Exception as e:
            print(f"Error with category page: {url}")
            print(e)


def id_generator(tourn_dict):
    """from tourney dict of form {year:[Tournaments]}, outputs tid_dict
    of form {tourney_id : Tournament} and defines Tournament.year
    """
    tid_dict = {}

    for year in tourn_dict:
        for tourn in tourn_dict[year]:
            tourn.year = year
            tid_dict.update({tourn.tourn_id: tourn})
    return tid_dict


def judge_writer(judges_dict, years, filename):
    """write csv with ballots per side per year per judge"""
    # get column names, incl side per year columns
    sides_list = []
    for year in years:
        sides_list.extend([f"{year} Aff Ballots", f"{year} Neg Ballots"])
    column_names = [
        "Judge ID",
        "Judge Name",
        "Judge's CEDA rounds",
        "Judge's Total CEDA Aff Rounds",
        "Judge's Total CEDA Neg Rounds",
        "Paradigm",
    ] + sides_list
    # iterate through judges, collecting data
    rows = []
    for judge in judges_dict.values():
        judge.side_count()
        output_row = [
            judge.id_num,
            judge.name,
            judge.ceda_rounds,
            judge.ceda_side_counter["Aff Total"],
            judge.ceda_side_counter["Neg Total"],
            judge.para,
        ]
        for year in years:
            output_row.extend(
                [
                    judge.ceda_side_counter[f"{year} Aff"],
                    judge.ceda_side_counter[f"{year} Neg"],
                ]
            )
        rows.append(output_row)
    paradigms = pd.DataFrame(rows, columns=column_names)
    # filter duplicates, keeping entry with more rounds
    # this affects <30 judges but can result in seeind old paradigms
    paradigms["Judge Name"] = paradigms["Judge Name"].apply(lambda x: str(x).lower())
    paradigms = paradigms.sort_values(
        by=["Judge's CEDA rounds", "Judge Name"], ascending=[False, True]
    )
    paradigms = paradigms.drop_duplicates(subset=["Judge Name"], keep="first")
    paradigms = paradigms[paradigms["Judge's CEDA rounds"] > 0]
    paradigms = paradigms.reset_index().drop(columns="index")

    paradigms.to_excel(filename)


def main(input_file, judge_out, results_out):
    """load list of relevant tournies from file, write file with
    judge info and file with round results"""
    print(f"Starting main function. Time is now {datetime.datetime.now()}\n")

    judge_dict = {}

    print(f"Gathering tournament data. Time is now {datetime.datetime.now()}\n")
    # load dictionary {year : [Tournament_0, ..., Tournament_n]}
    tourn_dict = load_tourney_list(input_file)

    # create dictionary {tourn_id : Tournament}
    tid_dict = id_generator(tourn_dict)

    print(f"Starting judge crawler. Time is now {datetime.datetime.now()}")
    for year, tourn_list in tqdm(tourn_dict.items(), desc="year", position=0):
        print(f"Starting year: {year}! Time is now {datetime.datetime.now()}")
        for tourn in tqdm(tourn_list, desc="tournaments", position=1, leave=False):
            judge_list = judge_cat_finder(tourn.judge_url)
            judge_info_finder(judge_list, judge_dict, tid_dict, tourn)

    print(f"All done with tournament collection! Time is now {datetime.datetime.now()}")
    print(f"Lenth of judge dict: {len(judge_dict)}")
    print("Now trying to write the data!")

    try:
        judge_writer(judge_dict, tourn_dict, judge_out)
    except Exception as e:
        print(e)

    jdl = list(judge_dict.values())
    # form massive dataframe with all round results
    column_names_part = [
        "ID",
        "season",
        "tourn",
        "?",
        "date",
        "division",
        "round",
        "aff",
        "neg",
        "ballot",
        "panel",
        "aff_id",
        "neg_id",
    ]
    column_names = [
        "ID",
        "season",
        "tourn",
        "?",
        "date",
        "division",
        "round",
        "aff",
        "neg",
        "ballot",
        "panel",
        "aff_id",
        "neg_id",
        "name",
    ]
    # initalize huge df with all round results
    the_big_one = pd.DataFrame(columns=column_names)
    for judge in jdl:
        to_add = pd.DataFrame(judge.ceda_round_list, columns=column_names_part)
        to_add["name"] = judge.name
        to_add["j_id"] = judge.id_num
        the_big_one = pd.concat([the_big_one, to_add], ignore_index=True)

    pools_dict = the_big_one.groupby("ID")["name"].unique().to_dict()
    the_big_one["pool"] = the_big_one["ID"].apply(lambda x: pools_dict[x].tolist())
    the_big_one.to_excel(results_out)
    print(f"ALL DONE! Time is now: {datetime.datetime.now()}")


if __name__ == "__main__":
    main(args.tourn_list, args.judge_output, args.round_output)
