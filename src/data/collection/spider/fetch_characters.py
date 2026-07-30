#!/usr/bin/env python3
"""
 ()
Fandom

"""
import os
import requests
import re
import time
import urllib.parse
from bs4 import BeautifulSoup
from concurrent.futures import ThreadPoolExecutor, as_completed

# 
HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
}


def clean_character_name(name):
    """"""
    if not name:
        return None

    # 
    name = re.sub(r"\(.*?\)", "", name)
    name = re.sub(r".*?", "", name)
    name = re.sub(r"\[.*?\]", "", name)

    # 
    name = name.strip()

    # 
    if len(name) < 2 or len(name) > 20:
        return None

    # 
    exclude_words = [
        "",
        "",
        "",
        "",
        "CV",
        "",
        "",
        "",
        "",
        "",
        "",
        "",
        "",
        "",
        "",
        "",
        "",
        "",
        "",
        "",
        "",
    ]
    if any(word in name for word in exclude_words):
        return None

    # 
    if re.match(r"^[\d\W]+$", name):
        return None

    return name


def fetch_url(url, retries=3):
    """"""
    for i in range(retries):
        try:
            response = requests.get(url, headers=HEADERS, timeout=10)
            if response.status_code == 200:
                return response
        except requests.RequestException:
            time.sleep(1)
    return None


def fetch_from_moegirl(anime_name):
    """"""
    print(f"  [] : {anime_name}")
    characters = set()

    try:
        # 1.  "/" 
        sub_page_url = f"https://zh.moegirl.org.cn/{urllib.parse.quote(anime_name)}/"
        response = fetch_url(sub_page_url)

        if response:
            print(f"  [] : {sub_page_url}")
            soup = BeautifulSoup(response.text, "html.parser")
            #  h2/h3 
            for link in soup.select("div.mw-parser-output a"):
                name = clean_character_name(link.text)
                if name:
                    characters.add(name)

        # 2. 
        if len(characters) < 5:
            search_url = f"https://zh.moegirl.org.cn/index.php?search={urllib.parse.quote(anime_name)}&title=Special:%E6%90%9C%E7%B4%A2&profile=default&fulltext=1"
            response = fetch_url(search_url)

            if response:
                soup = BeautifulSoup(response.text, "html.parser")
                page_url = None
                search_results = soup.select(".mw-search-result-heading a")
                if search_results:
                    page_url = "https://zh.moegirl.org.cn" + search_results[0].get("href")
                elif "" not in soup.title.string:
                    page_url = response.url

                if page_url:
                    print(f"  [] : {page_url}")
                    response = fetch_url(page_url)
                    if response:
                        soup = BeautifulSoup(response.text, "html.parser")

                        # A:  navbox ()
                        navboxes = soup.select("table.navbox")
                        for navbox in navboxes:
                            if any(k in navbox.text for k in ["", "", ""]):
                                for link in navbox.select("a"):
                                    name = clean_character_name(link.text)
                                    if name:
                                        characters.add(name)

                        # B: 
                        headers = soup.find_all(["h2", "h3"], string=re.compile(r"||"))
                        for header in headers:
                            for sibling in header.next_siblings:
                                if sibling.name in ["h2", "h3"]:
                                    break
                                if sibling.name in ["ul", "ol", "div", "table"]:
                                    if hasattr(sibling, "select"):
                                        for link in sibling.select("a"):
                                            name = clean_character_name(link.text)
                                            if name:
                                                characters.add(name)

    except Exception as e:
        print(f"  [] : {e}")

    return characters


def fetch_from_baike(anime_name):
    """"""
    print(f"  [] : {anime_name}")
    characters = set()

    try:
        url = f"https://baike.baidu.com/item/{urllib.parse.quote(anime_name)}"
        response = fetch_url(url)

        if response:
            soup = BeautifulSoup(response.text, "html.parser")

            # 1:  dt ()  b ()
            candidates = soup.select("dt, b, a")
            for cand in candidates:
                parent = cand.find_parent()
                if parent and ("" in parent.text or "" in parent.text):
                    name = clean_character_name(cand.text)
                    if name:
                        characters.add(name)

            # 2: 
            tables = soup.select("table")
            for table in tables:
                if "" in table.text or "" in table.text:
                    for row in table.select("tr"):
                        cols = row.select("td")
                        if cols:
                            name1 = clean_character_name(cols[0].text)
                            if name1:
                                characters.add(name1)
                            if len(cols) > 1:
                                name2 = clean_character_name(cols[1].text)
                                if name2:
                                    characters.add(name2)

    except Exception as e:
        print(f"  [] : {e}")

    return characters


def fetch_from_fandom(anime_name):
    """Fandom (IP)"""
    print(f"  [Fandom] : {anime_name}")
    characters = set()

    try:
        if "" in anime_name:
            url = "https://genshin-impact.fandom.com/wiki/Characters"
            print(f"  [Fandom] Wiki: {url}")
            response = fetch_url(url)
            if response:
                soup = BeautifulSoup(response.text, "html.parser")
                for link in soup.select("table.article-table td:nth-of-type(2) a"):
                    name = clean_character_name(link.text)
                    if name:
                        characters.add(name)

        elif "" in anime_name and "" in anime_name:
            url = "https://honkai-star-rail.fandom.com/wiki/Characters"
            print(f"  [Fandom] Wiki: {url}")
            response = fetch_url(url)
            if response:
                soup = BeautifulSoup(response.text, "html.parser")
                for link in soup.select("div.character-icon a"):
                    name = clean_character_name(link.get("title", ""))
                    if name:
                        characters.add(name)

        elif "" in anime_name:
            url = "https://wutheringwaves.fandom.com/wiki/Resonators"
            print(f"  [Fandom] Wiki: {url}")
            response = fetch_url(url)
            if response:
                soup = BeautifulSoup(response.text, "html.parser")
                for link in soup.select("div.wds-tab__content a"):
                    name = clean_character_name(link.get("title", ""))
                    if name:
                        characters.add(name)

    except Exception as e:
        print(f"  [Fandom] : {e}")

    return characters


def process_anime(anime, output_dir):
    """"""
    print(f"\n=== : {anime} ===")
    all_characters = set()

    # 
    with ThreadPoolExecutor(max_workers=3) as executor:
        futures = {
            executor.submit(fetch_from_moegirl, anime): "Moegirl",
            executor.submit(fetch_from_baike, anime): "Baike",
            executor.submit(fetch_from_fandom, anime): "Fandom",
        }

        for future in as_completed(futures):
            source = futures[future]
            try:
                chars = future.result()
                print(f"  > {source}  {len(chars)} ")
                all_characters.update(chars)
            except Exception as e:
                print(f"  > {source} : {e}")

    # 
    if all_characters:
        sorted_chars = sorted(list(all_characters))
        print(f"[{anime}]  {len(sorted_chars)} ")

        safe_filename = re.sub(r'[\\/*?:"<>|]', "_", anime)
        output_path = os.path.join(output_dir, f"{safe_filename}.txt")

        with open(output_path, "w", encoding="utf-8") as f:
            for char in sorted_chars:
                f.write(f"{char}\n")
        print(f": {output_path}")
    else:
        print(f"[{anime}] ")


def main():
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    input_file = os.path.join(base_dir, "anime_set.txt")
    output_dir = os.path.join(base_dir, "characters")

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    if not os.path.exists(input_file):
        # 
        if os.path.exists("auto_spider_img/anime_set.txt"):
            input_file = "auto_spider_img/anime_set.txt"
            output_dir = "auto_spider_img/characters"
        else:
            print(f": {input_file}")
            return

    with open(input_file, "r", encoding="utf-8") as f:
        anime_list = [line.strip() for line in f if line.strip()]

    # 
    # 
    for anime in anime_list:
        process_anime(anime, output_dir)
        time.sleep(1)  # 


if __name__ == "__main__":
    main()
