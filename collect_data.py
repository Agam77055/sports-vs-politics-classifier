"""
collect_data.py — Download and prepare the Sports vs Politics dataset.

Uses the BBC News dataset (public) and supplements with AG News-style
curated articles to build a balanced dataset.

Saves the result to data/dataset.csv with columns: text, label
"""

import csv
import os
import urllib.request
import tarfile
import shutil

DATA_DIR = os.path.join(os.path.dirname(__file__), "data")
BBC_URL = "http://mlg.ucd.ie/files/datasets/bbc-fulltext.zip"


def download_bbc():
    """Download and extract the BBC News dataset."""
    zip_path = os.path.join(DATA_DIR, "bbc-fulltext.zip")
    extract_dir = os.path.join(DATA_DIR, "bbc")

    if os.path.exists(extract_dir):
        print("BBC dataset already extracted.")
        return extract_dir

    print("Downloading BBC News dataset...")
    urllib.request.urlretrieve(BBC_URL, zip_path)
    print("Extracting...")

    import zipfile
    with zipfile.ZipFile(zip_path, 'r') as z:
        z.extractall(DATA_DIR)

    os.remove(zip_path)
    return extract_dir


def load_bbc_category(base_dir, category):
    """Load all articles from a BBC category folder."""
    cat_dir = os.path.join(base_dir, category)
    articles = []
    if not os.path.exists(cat_dir):
        print(f"  Warning: {cat_dir} not found")
        return articles
    for fname in os.listdir(cat_dir):
        fpath = os.path.join(cat_dir, fname)
        if os.path.isfile(fpath):
            with open(fpath, "r", encoding="utf-8", errors="ignore") as f:
                text = f.read().strip()
                # BBC files have title on first line, body after
                # combine them into one text block
                if text:
                    articles.append(text)
    return articles


def get_curated_sports():
    """Additional sports articles to supplement the dataset."""
    return [
        "The FIFA World Cup is the most watched sporting event globally with billions of viewers tuning in every four years. National teams compete in a month-long tournament that captivates fans across every continent.",
        "LeBron James scored 35 points in last night's NBA game leading the Lakers to a crucial victory against the Celtics in overtime. His performance was described as one of his best this season.",
        "Serena Williams announced her return to professional tennis after a two-year hiatus. The 23-time Grand Slam champion says she is motivated to compete at Wimbledon this summer.",
        "The Olympic Games committee announced new sports for the upcoming Summer Olympics including breakdancing and surfing. Athletes from over 200 countries are expected to compete.",
        "Manchester United completed a record transfer deal worth 100 million euros for the young Brazilian striker. The club hopes the signing will strengthen their attacking lineup for the Champions League.",
        "Usain Bolt's world record in the 100 meters still stands after more than a decade. Track and field analysts believe it may not be broken for another generation.",
        "The Indian Premier League cricket tournament saw record viewership this season with over 500 million viewers worldwide. The final between Mumbai Indians and Chennai Super Kings was the most watched match.",
        "Tiger Woods made his comeback at the Masters tournament at Augusta National. Golf fans around the world cheered as he completed all four rounds despite his recent injuries.",
        "Formula One racing saw a dramatic finish at the Monaco Grand Prix as the championship leader crashed on the final lap allowing his rival to take the victory.",
        "The NFL draft picked the top college quarterback as the first overall selection. Teams spent months scouting players and analyzing game footage to make their choices.",
        "Roger Federer retired from professional tennis after a legendary career spanning over two decades. He won 20 Grand Slam titles and is considered one of the greatest athletes of all time.",
        "The Tour de France cycling race covered over 3500 kilometers across the French countryside. Riders endured extreme heat and mountain stages to compete for the yellow jersey.",
        "Basketball star Stephen Curry broke the all-time three-point record during a regular season game. His shooting ability has revolutionized how the game of basketball is played.",
        "The rugby World Cup final was decided in extra time with a dramatic drop goal. New Zealand defeated South Africa in front of 80000 spectators at the stadium.",
        "Swimming champion Michael Phelps holds the record for most Olympic gold medals with 23. His dominance in the pool across four Olympic Games is unmatched in sporting history.",
        "The Boston Marathon is one of the oldest annual marathons in the world attracting elite runners from around the globe. Thousands of spectators line the streets to cheer on the participants.",
        "Cricket's Ashes series between England and Australia is one of the most intense rivalries in sport. The test matches are played over five days and demand incredible endurance from players.",
        "The Super Bowl halftime show has become almost as popular as the football game itself. Millions tune in specifically for the musical performances and advertisements.",
        "Novak Djokovic won his record-breaking Grand Slam title at the Australian Open. The Serbian tennis player has dominated the sport for over a decade.",
        "The Women's World Cup soccer tournament continues to grow in popularity and viewership every four years. National women's teams compete at the highest level of the sport.",
    ]


def get_curated_politics():
    """Additional politics articles to supplement the dataset."""
    return [
        "The United Nations General Assembly voted on a resolution regarding climate change policy. Delegates from 193 member nations debated the terms of the agreement for several days before reaching a consensus.",
        "The presidential election results were announced after a record voter turnout across the country. The winning candidate promised to focus on healthcare reform and economic recovery.",
        "Parliament passed a new immigration bill after months of heated debate between opposition parties. The legislation introduces stricter border controls and a points-based visa system.",
        "The prime minister held a press conference to address the ongoing economic crisis. New fiscal policies were announced including tax cuts for small businesses and increased public spending.",
        "Trade negotiations between the European Union and China reached a critical stage. Both sides are trying to resolve disputes over tariffs and intellectual property rights.",
        "The supreme court ruled on a landmark case regarding freedom of speech and digital privacy. The decision will have far-reaching implications for how social media companies operate.",
        "The government announced a new defense budget increasing military spending by fifteen percent. Critics argue the money would be better spent on education and healthcare.",
        "Foreign ministers from the G7 nations met to discuss the humanitarian crisis in the Middle East. Aid packages and diplomatic solutions were proposed during the two-day summit.",
        "The opposition leader called for a vote of no confidence in the ruling government citing corruption allegations. Protests erupted outside parliament as citizens demanded accountability.",
        "A bipartisan committee released its report on election security recommending new measures to protect voting systems from cyber attacks. Both parties agreed on the need for immediate action.",
        "The state governor signed an executive order banning certain types of assault weapons. Gun control advocates celebrated while opponents vowed to challenge the order in court.",
        "Diplomatic relations between Russia and Western nations deteriorated further following sanctions imposed over territorial disputes. Ambassadors were recalled from several countries.",
        "The national budget debate continued in congress with disagreements over healthcare funding and infrastructure spending. Both parties accused each other of fiscal irresponsibility.",
        "Municipal elections showed a shift in voter preferences toward progressive candidates in urban areas. Analysts attribute this to concerns about housing affordability and public transportation.",
        "The foreign affairs minister traveled to Africa for a diplomatic tour visiting five countries. Trade agreements and development aid were the primary topics of discussion.",
        "A new political party was formed by former members of the ruling coalition who disagreed with recent policy decisions. The party aims to provide a centrist alternative in the next election.",
        "The senate confirmed the new attorney general after a contentious confirmation hearing. Questions focused on the nominee's views on civil rights and criminal justice reform.",
        "International sanctions were imposed on the regime following evidence of human rights violations. The United Nations Security Council passed the resolution with a majority vote.",
        "The education minister proposed sweeping reforms to the national curriculum. The changes aim to modernize teaching methods and incorporate technology into classrooms across the country.",
        "Campaign finance reform legislation was introduced in parliament to limit corporate donations to political parties. Supporters say it will reduce corruption while opponents claim it restricts free speech.",
    ]


def build_dataset():
    """Build the combined dataset from BBC + curated articles."""
    os.makedirs(DATA_DIR, exist_ok=True)

    rows = []

    # --- BBC dataset ---
    try:
        bbc_dir = download_bbc()
        sport_articles = load_bbc_category(bbc_dir, "sport")
        politics_articles = load_bbc_category(bbc_dir, "politics")
        print(f"BBC Sport articles: {len(sport_articles)}")
        print(f"BBC Politics articles: {len(politics_articles)}")

        for text in sport_articles:
            rows.append({"text": text, "label": "sports"})
        for text in politics_articles:
            rows.append({"text": text, "label": "politics"})
    except Exception as e:
        print(f"Could not download BBC dataset: {e}")
        print("Proceeding with curated data only.")

    # --- Curated supplementary articles ---
    for text in get_curated_sports():
        rows.append({"text": text, "label": "sports"})
    for text in get_curated_politics():
        rows.append({"text": text, "label": "politics"})

    # --- Save to CSV ---
    csv_path = os.path.join(DATA_DIR, "dataset.csv")
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["text", "label"])
        writer.writeheader()
        for row in rows:
            writer.writerow(row)

    # count per class
    sports_count = sum(1 for r in rows if r["label"] == "sports")
    politics_count = sum(1 for r in rows if r["label"] == "politics")
    print(f"\nDataset saved to {csv_path}")
    print(f"  Sports: {sports_count}")
    print(f"  Politics: {politics_count}")
    print(f"  Total: {len(rows)}")

    # cleanup extracted BBC folder
    bbc_extracted = os.path.join(DATA_DIR, "bbc")
    if os.path.exists(bbc_extracted):
        shutil.rmtree(bbc_extracted)
        print("Cleaned up extracted BBC files.")


if __name__ == "__main__":
    build_dataset()
