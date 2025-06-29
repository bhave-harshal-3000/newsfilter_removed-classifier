import os
import time
import requests
from bs4 import BeautifulSoup
from dotenv import load_dotenv
import smtplib
from email.message import EmailMessage
from langchain.chat_models import init_chat_model
from langchain_core.messages import HumanMessage

load_dotenv()
os.environ["GOOGLE_API_KEY"] = os.getenv("GOOGLE_API_KEY")

headers = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"
}

all_headlines = []

# 1. The Hindu
print("Scraping The Hindu...")
for page in range(1, 4):
    url = f"https://www.thehindu.com/education/?page={page}"
    print(f"Fetching {url}")
    try:
        response = requests.get(url, headers=headers)
        soup = BeautifulSoup(response.text, "html.parser")
        titles = soup.select("h3.title > a") + soup.select("h3.title.big > a")
        for t in titles:
            headline = t.get_text(strip=True)
            link = t.get("href")
            if link and not link.startswith("http"):
                link = "https://www.thehindu.com" + link
            all_headlines.append(["The Hindu", headline, link])
    except Exception as e:
        print(f"Error on page {page}: {e}")
    time.sleep(1)

# 2. Hindustan Times
print("Scraping Hindustan Times...")
ht_url = "https://www.hindustantimes.com/topic/education-news/news"
try:
    response = requests.get(ht_url, headers=headers)
    soup = BeautifulSoup(response.text, "html.parser")
    titles = soup.select("h3.hdg3")
    for t in titles:
        headline = t.get_text(strip=True)
        link_tag = t.find("a")
        link = "https://www.hindustantimes.com" + link_tag.get("href") if link_tag else ""
        all_headlines.append(["Hindustan Times", headline, link])
except Exception as e:
    print(f"Error scraping HT: {e}")

# 3. Times of India
print("Scraping Times of India...")
toi_url = "https://timesofindia.indiatimes.com/education"
try:
    response = requests.get(toi_url, headers=headers)
    soup = BeautifulSoup(response.text, "html.parser")
    figures = soup.find_all("figure")
    for fig in figures:
        a_tag = fig.find("a")
        if a_tag:
            headline = a_tag.get_text(strip=True)
            link = a_tag.get("href")
            if link and not link.startswith("http"):
                link = "https://timesofindia.indiatimes.com" + link
            all_headlines.append(["Times of India", headline, link])
except Exception as e:
    print(f"Error scraping TOI: {e}")

# 4. Indian Express
print("Scraping Indian Express...")
ie_url = "https://indianexpress.com/section/education/"
try:
    response = requests.get(ie_url, headers=headers)
    soup = BeautifulSoup(response.text, "html.parser")
    articles = soup.select("div.articles h2 a")
    for tag in articles:
        headline = tag.get_text(strip=True)
        link = tag.get("href")
        all_headlines.append(["Indian Express", headline, link])
except Exception as e:
    print(f"Error scraping Indian Express: {e}")



# 5. BBC Future (Education)
print("Scraping BBC News...")
bbc_url = "https://www.bbc.com/future/tags/education"
try:
    response = requests.get(bbc_url, headers=headers)
    soup = BeautifulSoup(response.text, "html.parser")
    # Find all h2 tags with data-testid="card-headline"
    headlines = soup.find_all("h2", attrs={"data-testid": "card-headline"})
    for h in headlines:
        headline = h.get_text(strip=True)
        # Find the closest ancestor <a> tag for the link
        a_tag = h.find_parent("a")
        link = a_tag.get("href") if a_tag else ""
        if link and not link.startswith("http"):
            link = "https://www.bbc.com" + link
        all_headlines.append(["BBC Future", headline, link])
except Exception as e:
    print(f"Error scraping BBC Future: {e}")

print(f"Total headlines scraped: {len(all_headlines)}")

# 6. LLM Deduplication + Classification
model = init_chat_model("gemini-2.0-flash", model_provider="google_genai")
prompt = (
    "You are an expert news assistant. "
    "Given the following list of news headlines (with their sources and links), "
    "deduplicate them by grouping together headlines that refer to the same news event, "
    "even if they are worded differently or come from different sources. "
    "For each unique news event, pick the most informative headline (with its source and link). "
    "Then, classify each unique headline as either 'INDIA' or 'GLOBAL'.\n\n"
    "Return your answer as a numbered list in this format:\n"
    "<SOURCE>, <HEADLINE>\n<LINK>\n<INDIA/GLOBAL>\n\n"
    "Here is the list:\n"
)
for idx, (source, headline, link) in enumerate(all_headlines, 1):
    prompt += f"{idx}. {source}, {headline}\n{link}\n"

response = model.invoke([HumanMessage(content=prompt)])



# 7. Parse LLM result, skipping intro lines
india_entries = []
global_entries = []
lines = response.content.split('\n')

# Skip lines until we reach the first numbered headline
while lines and not (lines[0].strip() and lines[0].strip()[0].isdigit() and '.' in lines[0]):
    lines.pop(0)

i = 0
while i < len(lines):
    line = lines[i].strip()
    if line and ',' in line and line[0].isdigit() and '.' in line:
        source_headline = line.split('.', 1)[1].strip()  # Remove the number
        link = lines[i+1].strip() if i+1 < len(lines) else ""
        label = lines[i+2].strip().upper() if i+2 < len(lines) else ""
        entry = f"{source_headline}\n{link}"
        if "INDIA" in label:
            india_entries.append(entry)
        elif "GLOBAL" in label:
            global_entries.append(entry)
        i += 3
    else:
        i += 1

def build_email_content(entries, label):
    if not entries:
        return f"No {label} headlines found."
    return "\n\n".join([f"{idx+1}. {entry}" for idx, entry in enumerate(entries)])

india_content = build_email_content(india_entries, "INDIA")
global_content = build_email_content(global_entries, "GLOBAL")

# 8. Send emails (same as before)
EMAIL_USER = os.getenv('EMAIL_USER')
EMAIL_PASS = os.getenv('EMAIL_PASS')
EMAIL_TO = 'harshalbhave05@gmail.com'  # Change as needed

def send_email(subject, content):
    msg = EmailMessage()
    msg['Subject'] = subject
    msg['From'] = EMAIL_USER
    msg['To'] = EMAIL_TO
    msg.set_content(content)
    try:
        with smtplib.SMTP_SSL('smtp.gmail.com', 465) as smtp:
            smtp.login(EMAIL_USER, EMAIL_PASS)
            smtp.send_message(msg)
        print(f"Email sent: {subject}")
    except Exception as e:
        print(f"Failed to send email '{subject}': {e}")

send_email("INDIA Education News Headlines", india_content.strip())
send_email("GLOBAL Education News Headlines", global_content.strip())